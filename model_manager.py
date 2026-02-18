# model_manager.py - 記憶體感知模型管理器（支援多模型）

import os
import time
import threading
import gc
import uuid
from collections import deque
from typing import Optional, Dict, List, Any, Tuple

import openvino_genai as ov_genai

# 嘗試導入 psutil 用於記憶體監控
try:
    import psutil
    import humanize
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    def format_memory_bytes(bytes_val):
        return f"{bytes_val:,} bytes"

def get_memory_usage():
    if not HAS_PSUTIL:
        return {
            "process_rss": 0, "process_vms": 0,
            "system_total": 0, "system_available": 0,
            "system_used": 0, "system_percent": 0
        }
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        return {
            "process_rss": memory_info.rss,
            "process_vms": memory_info.vms,
            "system_total": system_memory.total,
            "system_available": system_memory.available,
            "system_used": system_memory.used,
            "system_percent": system_memory.percent
        }
    except Exception:
        return {
            "process_rss": 0, "process_vms": 0,
            "system_total": 0, "system_available": 0,
            "system_used": 0, "system_percent": 0
        }

def format_memory_bytes(bytes_val):
    if not HAS_PSUTIL:
        return f"{bytes_val:,} bytes"
    try:
        return humanize.naturalsize(bytes_val)
    except:
        return f"{bytes_val:,} bytes"

def print_memory_usage(label="當前", device_name=""):
    mem = get_memory_usage()
    prefix = f"[{device_name}]" if device_name else ""
    print(f"\n[記憶體] {prefix} {label} 記憶體使用:")
    print(f"   進程 RSS: {format_memory_bytes(mem['process_rss'])}")
    print(f"   進程 VMS: {format_memory_bytes(mem['process_vms'])}")
    print(f"   系統使用: {mem['system_percent']:.1f}% ({format_memory_bytes(mem['system_used'])} / {format_memory_bytes(mem['system_total'])})")
    return mem


class MemoryAwareModelManager:
    """單一模型的記憶體感知管理器（供 MultiModelManager 內部使用）"""

    def __init__(self, model_config: dict, device_name: str, device_type: str, cache_suffix: str):
        self.model_id = model_config["id"]
        self.model_path = model_config["path"]
        self.device_name = device_name
        self.device_type = device_type
        self.cache_suffix = cache_suffix

        self.estimated_weight_memory_gb = model_config.get("estimated_memory_gb", 6.0)
        self.estimated_instance_memory_gb = model_config.get("instance_memory_gb", 2.0)
        self.max_total_memory_gb = model_config.get("max_total_memory_gb", 64.0)
        self.max_instances = model_config.get("max_concurrent", 100)

        self._reserved_instances = 0
        self._reserved_memory_gb = 0.0
        self._lock = threading.RLock()
        self.instance_available = threading.Condition(self._lock)

        self.idle_timeout_seconds = model_config.get("idle_timeout", 300)
        self.extra_config = model_config.get("extra_config", {})

        self.model_pool = deque()
        self.active_models = {}
        self.model_counter = 0
        self.total_created = 0

        # 快取路徑可依模型不同區分
        self.cache_path = f"./cache_{self.cache_suffix}_{self.model_id.replace('/', '_')}"
        os.makedirs(self.cache_path, exist_ok=True)

        self.running = True
        self.memory_monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self.idle_monitor_thread = threading.Thread(target=self._idle_instance_monitor_loop, daemon=True)
        self.memory_monitor_thread.start()
        self.idle_monitor_thread.start()

        self.stats = {
            "instances_created": 0,
            "instances_destroyed": 0,
            "requests_served": 0,
            "memory_checks_failed": 0,
            "peak_memory_usage_gb": 0.0,
            "timeout_count": 0,
            "reserve_failures": 0,
        }

        print(f"[管理器] [{self.device_name}:{self.model_id}] 初始化完成")
        print(f"   - 最大實例數: {self.max_instances}")
        print(f"   - 預估實例記憶體: {self.estimated_instance_memory_gb} GB")
        print(f"   - 最大總記憶體: {self.max_total_memory_gb} GB")
        print(f"   - 快取路徑: {self.cache_path}")

    # ------------------------------ 內部監控 ------------------------------
    def _memory_monitor_loop(self):
        while self.running:
            try:
                self._check_memory_and_adjust()
                time.sleep(15)
            except Exception:
                time.sleep(30)

    def _idle_instance_monitor_loop(self):
        while self.running:
            try:
                self._cleanup_idle_instances()
                time.sleep(60)
            except Exception:
                time.sleep(60)

    def _check_memory_and_adjust(self):
        if not HAS_PSUTIL:
            return
        with self._lock:
            mem = get_memory_usage()
            used_gb = mem["process_rss"] / (1024**3)
            if used_gb > self.stats["peak_memory_usage_gb"]:
                self.stats["peak_memory_usage_gb"] = used_gb

            total_instances = len(self.model_pool) + len(self.active_models) + self._reserved_instances
            available_gb = self.max_total_memory_gb - used_gb
            if available_gb < self.estimated_instance_memory_gb * 0.8:
                safe_max = max(1, total_instances - 1)
                if self.max_instances > safe_max:
                    self.max_instances = safe_max
                    print(f"[警告] [{self.device_name}:{self.model_id}] 記憶體緊張，限制最大實例數為: {safe_max}")
            elif available_gb > self.estimated_instance_memory_gb * 2:
                original_max = self.extra_config.get("max_concurrent", 100)
                if self.max_instances < original_max:
                    self.max_instances = original_max
                    print(f"[資訊] [{self.device_name}:{self.model_id}] 記憶體充足，恢復最大實例數為: {original_max}")

    def _cleanup_idle_instances(self):
        with self._lock:
            now = time.time()
            to_remove = []
            for i, (inst, last) in enumerate(self.model_pool):
                if now - last > self.idle_timeout_seconds:
                    to_remove.append(i)
            if to_remove and len(self.model_pool) > 1:
                for i in reversed(to_remove):
                    inst, _ = self.model_pool[i]
                    del self.model_pool[i]
                    self._destroy_instance(inst)

    def _destroy_instance(self, instance):
        try:
            if "pipe" in instance:
                del instance["pipe"]
            if "tokenizer" in instance:
                del instance["tokenizer"]
            self.stats["instances_destroyed"] += 1
            print(f"[銷毀] [{self.device_name}:{self.model_id}] 實例 {instance['id']} 已銷毀")
        except Exception as e:
            print(f"[錯誤] [{self.device_name}:{self.model_id}] 銷毀實例失敗: {e}")
        finally:
            gc.collect()

    # ------------------------------ 實例建立 ------------------------------
    def create_model_instance(self, instance_id):
        with self._lock:
            total_instances = len(self.model_pool) + len(self.active_models) + self._reserved_instances
            if total_instances >= self.max_instances:
                print(f"[錯誤] [{self.device_name}:{self.model_id}] 已達最大實例數，無法創建")
                return None
            estimated_total = (total_instances + 1) * self.estimated_instance_memory_gb
            if estimated_total > self.max_total_memory_gb:
                print(f"[錯誤] [{self.device_name}:{self.model_id}] 預估記憶體不足，無法創建")
                self.stats["reserve_failures"] += 1
                return None
            self._reserved_instances += 1
            self._reserved_memory_gb += self.estimated_instance_memory_gb

        try:
            print(f"[載入] [{self.device_name}:{self.model_id}#{instance_id}] 正在載入模型到 {self.device_type}...")
            mem_before = print_memory_usage("載入前", f"{self.device_name}:{self.model_id}") if HAS_PSUTIL else None

            config = {"CACHE_DIR": self.cache_path}
            for k, v in self.extra_config.items():
                config[k] = v

            pipe = ov_genai.LLMPipeline(self.model_path, self.device_type, config)
            tokenizer_obj = ov_genai.Tokenizer(self.model_path)

            print(f"[成功] [{self.device_name}:{self.model_id}#{instance_id}] 模型載入成功！")
            if HAS_PSUTIL:
                mem_after = get_memory_usage()
                mem_increase = (mem_after["process_rss"] - mem_before["process_rss"]) / (1024**3)
                print(f"[記憶體] 實際記憶體增加: {mem_increase:.2f} GB")
            else:
                mem_increase = 0

            self.total_created += 1
            self.stats["instances_created"] += 1

            return {
                "id": instance_id,
                "pipe": pipe,
                "tokenizer": tokenizer_obj,
                "created_time": time.time(),
                "last_used_time": time.time(),
                "use_count": 0,
                "memory_increase_gb": mem_increase,
            }
        except Exception as e:
            print(f"[錯誤] [{self.device_name}:{self.model_id}] 模型載入失敗: {e}")
            try:
                print(f"[回退] [{self.device_name}:{self.model_id}] 嘗試使用 CPU 回退...")
                pipe = ov_genai.LLMPipeline(self.model_path, "CPU", {"CACHE_DIR": self.cache_path})
                tokenizer_obj = ov_genai.Tokenizer(self.model_path)
                print(f"[警告] [{self.device_name}:{self.model_id}] 使用 CPU 回退模式")
                return {
                    "id": f"{instance_id}-cpu",
                    "pipe": pipe,
                    "tokenizer": tokenizer_obj,
                    "created_time": time.time(),
                    "last_used_time": time.time(),
                    "use_count": 0,
                    "fallback": True,
                    "memory_increase_gb": 0,
                }
            except Exception as e2:
                print(f"[錯誤] [{self.device_name}:{self.model_id}] 所有設備載入都失敗: {e2}")
                return None
        finally:
            with self._lock:
                self._reserved_instances -= 1
                self._reserved_memory_gb -= self.estimated_instance_memory_gb

    # ------------------------------ 對外接口 ------------------------------
    def get_model_instance(self, timeout=30):
        start = time.time()
        request_id = str(uuid.uuid4())[:8]

        with self._lock:
            if self.model_pool:
                inst, _ = self.model_pool.popleft()
                inst["last_used_time"] = time.time()
                inst["use_count"] += 1
                self.active_models[request_id] = inst
                print(f"[獲取] [{self.device_name}:{self.model_id}#{request_id}] 獲取空閒實例 {inst['id']}")
                self._print_stats()
                return request_id, inst

            total = len(self.model_pool) + len(self.active_models) + self._reserved_instances
            if total < self.max_instances:
                if (total + 1) * self.estimated_instance_memory_gb <= self.max_total_memory_gb:
                    self.model_counter += 1
                    instance_id = f"inst-{self.model_counter:03d}"
                    self._lock.release()
                    try:
                        inst = self.create_model_instance(instance_id)
                    finally:
                        self._lock.acquire()
                    if inst:
                        inst["last_used_time"] = time.time()
                        inst["use_count"] = 1
                        self.active_models[request_id] = inst
                        print(f"[新增] [{self.device_name}:{self.model_id}#{request_id}] 創建新實例 {inst['id']}")
                        self._print_stats()
                        return request_id, inst
                else:
                    self.stats["reserve_failures"] += 1

        with self.instance_available:
            while time.time() - start < timeout:
                with self._lock:
                    if self.model_pool:
                        inst, _ = self.model_pool.popleft()
                        inst["last_used_time"] = time.time()
                        inst["use_count"] += 1
                        self.active_models[request_id] = inst
                        print(f"[等待] [{self.device_name}:{self.model_id}#{request_id}] 等待後獲取實例")
                        self._print_stats()
                        return request_id, inst
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    break
                self.instance_available.wait(timeout=min(1.0, remaining))

        self.stats["timeout_count"] += 1
        print(f"[超時] [{self.device_name}:{self.model_id}#{request_id}] 獲取實例超時")
        return None

    def release_model_instance(self, request_id):
        with self._lock:
            if request_id not in self.active_models:
                print(f"[警告] [{self.device_name}:{self.model_id}#{request_id}] 釋放失敗: 請求ID不存在")
                return False
            inst = self.active_models.pop(request_id)
            inst["last_used_time"] = time.time()
            self.model_pool.append((inst, inst["last_used_time"]))
            self.stats["requests_served"] += 1
            print(f"[釋放] [{self.device_name}:{self.model_id}#{request_id}] 實例 {inst['id']} 已釋放")
            self._print_stats()

        with self.instance_available:
            self.instance_available.notify_all()
        return True

    def _print_stats(self):
        total = len(self.model_pool) + len(self.active_models)
        print(f"[統計] [{self.device_name}:{self.model_id}] 實例狀態: {len(self.active_models)}使用中, {len(self.model_pool)}空閒, {total}總數")

    def preload_instances(self, count=1):
        success = 0
        for i in range(count):
            with self._lock:
                total = len(self.model_pool) + len(self.active_models) + self._reserved_instances
                if total >= self.max_instances:
                    break
            self.model_counter += 1
            inst_id = f"preload-{self.model_counter:03d}"
            inst = self.create_model_instance(inst_id)
            if inst:
                inst["last_used_time"] = time.time()
                with self._lock:
                    self.model_pool.append((inst, inst["last_used_time"]))
                success += 1
        return success

    def get_stats(self):
        with self._lock:
            instances_info = []
            for req_id, inst in self.active_models.items():
                instances_info.append({
                    "request_id": req_id,
                    "instance_id": inst["id"],
                    "use_count": inst["use_count"],
                    "created_time": inst["created_time"],
                    "last_used": inst["last_used_time"],
                    "status": "active",
                    "fallback": inst.get("fallback", False)
                })
            now = time.time()
            for inst, last_used in self.model_pool:
                instances_info.append({
                    "instance_id": inst["id"],
                    "use_count": inst["use_count"],
                    "created_time": inst["created_time"],
                    "last_used": last_used,
                    "idle_seconds": now - last_used,
                    "status": "idle",
                    "fallback": inst.get("fallback", False)
                })

            mem = get_memory_usage()
            used_gb = mem["process_rss"] / (1024**3) if HAS_PSUTIL else 0

            return {
                "model_id": self.model_id,
                "device_name": self.device_name,
                "device_type": self.device_type,
                "max_instances": self.max_instances,
                "active_instances": len(self.active_models),
                "idle_instances": len(self.model_pool),
                "total_instances": len(self.active_models) + len(self.model_pool),
                "reserved_instances": self._reserved_instances,
                "memory_used_gb": used_gb,
                "memory_max_gb": self.max_total_memory_gb,
                "peak_memory_gb": self.stats["peak_memory_usage_gb"],
                "instances_created": self.stats["instances_created"],
                "instances_destroyed": self.stats["instances_destroyed"],
                "requests_served": self.stats["requests_served"],
                "memory_checks_failed": self.stats["memory_checks_failed"],
                "reserve_failures": self.stats["reserve_failures"],
                "timeout_count": self.stats["timeout_count"],
                "estimated_instance_memory_gb": self.estimated_instance_memory_gb,
                "idle_timeout_seconds": self.idle_timeout_seconds,
                "instances": instances_info,
            }

    def shutdown(self):
        self.running = False
        self.memory_monitor_thread.join(timeout=5)
        self.idle_monitor_thread.join(timeout=5)
        print(f"[關閉] [{self.device_name}:{self.model_id}] 模型管理器已關閉")


class MultiModelManager:
    """管理多個模型的實例池，統一對外接口"""

    def __init__(self, device_config: dict):
        self.device_name = device_config["name"]
        self.device_type = device_config["device"]
        self.models: Dict[str, MemoryAwareModelManager] = {}

        for model_cfg in device_config.get("models", []):
            model_id = model_cfg["id"]
            self.models[model_id] = MemoryAwareModelManager(
                model_config=model_cfg,
                device_name=self.device_name,
                device_type=self.device_type,
                cache_suffix=self.device_name.lower()
            )
            # 可根據需要預載
            preload = model_cfg.get("preload_count", 0)
            if preload > 0:
                self.models[model_id].preload_instances(preload)

        print(f"[多模型管理器] [{self.device_name}] 已載入模型: {list(self.models.keys())}")

    def get_model_instance(self, model_id: str, timeout=30):
        if model_id not in self.models:
            raise ValueError(f"模型 {model_id} 不存在於設備 {self.device_name}")
        return self.models[model_id].get_model_instance(timeout)

    def release_model_instance(self, model_id: str, request_id: str):
        if model_id not in self.models:
            raise ValueError(f"模型 {model_id} 不存在")
        return self.models[model_id].release_model_instance(request_id)

    def get_stats(self, model_id: Optional[str] = None):
        if model_id:
            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 不存在")
            return self.models[model_id].get_stats()
        else:
            return {mid: mgr.get_stats() for mid, mgr in self.models.items()}

    def shutdown(self):
        for mgr in self.models.values():
            mgr.shutdown()