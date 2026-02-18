# server_app.py - FastAPI 應用程式（簡化 prompt，不處理歷史對話）

import asyncio
import concurrent.futures
import json
import time
import uuid
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from model_manager import (
    MultiModelManager,
    get_memory_usage,
    format_memory_bytes,
    print_memory_usage,
    HAS_PSUTIL
)

# ------------------------------------------------------------
# AsyncStreamer 類別
# ------------------------------------------------------------
class AsyncStreamer:
    def __init__(self, loop, request_id: str = "", model_instance_id: str = "", model_id: str = ""):
        self.queue = asyncio.Queue(maxsize=2000)
        self.loop = loop
        self.token_count = 0
        self.start_time = None
        self.request_id = request_id
        self.model_instance_id = model_instance_id
        self.model_id = model_id
        self.token_times = []
        self.finished = False
        self.generation_started = False
        self.generation_successful = False
        self.queue_full_count = 0

    def __call__(self, token):
        if self.finished:
            return False
        if self.start_time is None:
            self.start_time = time.time()
            self.generation_started = True
        self.token_count += 1
        self.token_times.append(time.time())
        future = asyncio.run_coroutine_threadsafe(self.queue.put(token), self.loop)
        try:
            future.result()
        except Exception as e:
            print(f"[警告] [{self.request_id}] 放入佇列失敗: {e}")
            return False
        return False

    def end(self, success=True):
        self.finished = True
        self.generation_successful = success
        asyncio.run_coroutine_threadsafe(self.queue.put(None), self.loop)

    def get_stats(self):
        if not self.token_times or self.token_count == 0 or not self.generation_started:
            return {"token_count": 0, "total_time": 0, "tokens_per_second": 0,
                    "generation_successful": self.generation_successful}
        total_time = self.token_times[-1] - self.start_time
        return {
            "token_count": self.token_count,
            "total_time": total_time,
            "tokens_per_second": self.token_count / total_time if total_time > 0 else 0,
            "generation_successful": self.generation_successful,
            "queue_full_count": self.queue_full_count,
        }

# ------------------------------------------------------------
# 安全的生成函數
# ------------------------------------------------------------
def safe_generate_with_cleanup(model_instance, prompt: str, streamer: AsyncStreamer,
                               request_id: str = "", multi_manager: MultiModelManager = None,
                               model_id: str = ""):
    device_name = multi_manager.device_name
    instance_id = model_instance.get("id", "unknown")
    try:
        print(f"[{device_name}:{model_id}#{request_id}|{instance_id}] 開始生成...")
        if HAS_PSUTIL:
            mem_before = get_memory_usage()
            print(f"[{device_name}:{model_id}#{request_id}] 生成前記憶體: {format_memory_bytes(mem_before['process_rss'])}")

        pipe = model_instance["pipe"]
        pipe.generate(prompt, max_new_tokens=1024, streamer=streamer)

        print(f"[{device_name}:{model_id}#{request_id}|{instance_id}] 生成完成")
        streamer.end(success=True)
    except Exception as e:
        print(f"\n[錯誤] [{device_name}:{model_id}#{request_id}|{instance_id}] 生成失敗: {e}")
        streamer.end(success=False)
        raise
    finally:
        if multi_manager:
            multi_manager.release_model_instance(model_id, request_id)
        if HAS_PSUTIL and 'mem_before' in locals():
            mem_after = get_memory_usage()
            increase = mem_after['process_rss'] - mem_before['process_rss']
            if increase > 100 * 1024 * 1024:
                print(f"[警告] [{device_name}:{model_id}#{request_id}] 生成過程記憶體顯著增加: {format_memory_bytes(increase)}")

# ------------------------------------------------------------
# 建立應用程式
# ------------------------------------------------------------
def create_app(device_config=None):
    """
    建立 FastAPI 應用。若 device_config 為 None，則從環境變數讀取配置。
    返回 ASGI 應用。
    """
    if device_config is None:
        config_json = os.environ.get("DEVICE_CONFIG_JSON")
        if not config_json:
            raise ValueError("未提供 device_config，且環境變數 DEVICE_CONFIG_JSON 為空")
        device_config = json.loads(config_json)

    # 為每個模型設定 extra_config（如 MAX_PROMPT_LEN 僅用於 NPU）
    for model_cfg in device_config.get("models", []):
        extra = model_cfg.get("extra_config", {})
        if device_config["name"] == "NPU":
            extra["MAX_PROMPT_LEN"] = model_cfg.get("max_prompt_len", 16000)
        else:
            extra["PERFORMANCE_HINT"] = "THROUGHPUT"
        model_cfg["extra_config"] = extra

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"[啟動] [{device_config['name']}] 啟動 lifespan，載入模型: {[m['id'] for m in device_config.get('models', [])]}")
        yield
        print(f"[關閉] [{device_config['name']}] 關閉 lifespan，釋放資源")
        thread_pool.shutdown(wait=True)
        multi_manager.shutdown()

    app = FastAPI(
        title=f"{device_config['name']} AI Server (Multi-Model)",
        lifespan=lifespan
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    multi_manager = MultiModelManager(device_config)
    device_prefix = device_config["name"].lower()  # "gpu" 或 "npu"

    # 建立複合 ID 與原始 ID 的對應（用於請求解析）
    model_id_map = {}
    for model_id in multi_manager.models.keys():
        composite_id = f"{device_prefix}-{model_id}"
        model_id_map[composite_id] = model_id

    thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=device_config.get("max_concurrent", 100),
        thread_name_prefix=f"{device_config['name']}_Worker"
    )

    # 輔助函數：將請求中的模型 ID 轉換為原始 ID
    def resolve_model_id(requested_id: str) -> str:
        """解析可能帶前綴的模型 ID，返回原始 ID；若前綴不符則拋出 400"""
        if not requested_id:
            return None
        if requested_id.startswith("gpu-") or requested_id.startswith("npu-"):
            prefix, original = requested_id.split("-", 1)
            if prefix != device_prefix:
                raise HTTPException(status_code=400, detail=f"模型 {requested_id} 不屬於當前設備 {device_config['name']}")
            return original
        # 向後相容：若無前綴，直接視為原始 ID（但建議客戶端更新）
        print(f"[警告] 收到不帶前綴的模型 ID '{requested_id}'，請考慮使用 '{device_prefix}-{requested_id}' 以明確指定設備")
        return requested_id

    # --------------------------------------------------------
    # 路由定義
    # --------------------------------------------------------
    @app.get("/health")
    async def health_check():
        stats = multi_manager.get_stats()
        mem_info = get_memory_usage()
        return {
            "status": "healthy",
            "device": device_config["name"],
            "device_type": device_config["device"],
            "port": device_config["port"],
            "timestamp": time.time(),
            "models": stats,
            "memory": {
                "process_rss_bytes": mem_info["process_rss"],
                "process_rss_human": format_memory_bytes(mem_info["process_rss"]),
                "system_used_percent": mem_info["system_percent"],
            },
        }

    @app.get("/v1/models")
    @app.get("/models")
    async def list_models():
        models_data = []
        for model_id, mgr_stats in multi_manager.get_stats().items():
            composite_id = f"{device_prefix}-{model_id}"
            models_data.append({
                "id": composite_id,  # 對外使用複合 ID
                "object": "model",
                "device": device_config["name"],
                "device_type": device_config["device"],
                "display_name": model_id,
                "port": device_config["port"],
                "max_context_length": mgr_stats.get("extra_config", {}).get("MAX_PROMPT_LEN", 16000),
                "instance_stats": {
                    "active": mgr_stats["active_instances"],
                    "idle": mgr_stats["idle_instances"],
                    "total": mgr_stats["total_instances"]
                }
            })
        return {
            "object": "list",
            "data": models_data
        }

    @app.get("/api/tags")
    async def ollama_tags():
        tags = []
        for model_id, mgr_stats in multi_manager.get_stats().items():
            composite_id = f"{device_prefix}-{model_id}"
            tags.append({
                "name": composite_id,
                "model": composite_id,
                "display_name": model_id,
                "details": {
                    "parameter_size": "未知",
                    "device": device_config["name"],
                    "device_type": device_config["device"],
                    "quantization": "INT4",
                    "port": device_config["port"],
                    "max_context_length": mgr_stats.get("extra_config", {}).get("MAX_PROMPT_LEN", 16000),
                    "max_instances": mgr_stats["max_instances"],
                    "current_instances": mgr_stats["total_instances"],
                }
            })
        return {"models": tags}

    @app.get("/memory")
    async def memory_info():
        mem_info = get_memory_usage()
        return {
            "device": device_config["name"],
            "process_memory": {
                "rss_bytes": mem_info["process_rss"],
                "rss_human": format_memory_bytes(mem_info["process_rss"]),
            },
            "system_memory": {
                "total_human": format_memory_bytes(mem_info["system_total"]),
                "available_human": format_memory_bytes(mem_info["system_available"]),
                "used_percent": mem_info["system_percent"],
            }
        }

    @app.get("/model/stats")
    async def model_stats(model: Optional[str] = None):
        if model:
            original = resolve_model_id(model)
            return multi_manager.get_stats(original)
        return multi_manager.get_stats()

    @app.get("/model/instances")
    async def list_instances(model: Optional[str] = None):
        if model:
            original = resolve_model_id(model)
            stats = multi_manager.get_stats(original)
            return {
                "device": device_config["name"],
                "model": original,
                "instances": stats.get("instances", []),
                "total_instances": stats["total_instances"],
                "active_instances": stats["active_instances"],
                "idle_instances": stats["idle_instances"]
            }
        else:
            result = {}
            for mid, stats in multi_manager.get_stats().items():
                result[mid] = {
                    "instances": stats.get("instances", []),
                    "total_instances": stats["total_instances"],
                    "active_instances": stats["active_instances"],
                    "idle_instances": stats["idle_instances"]
                }
            return result

    @app.post("/model/preload/{model_id}/{count}")
    async def preload_models(model_id: str, count: int = 1):
        if count < 1 or count > 10:
            raise HTTPException(status_code=400, detail="預載入數量必須在 1-10 之間")
        original = resolve_model_id(model_id)
        if original not in multi_manager.models:
            raise HTTPException(status_code=404, detail="模型不存在")
        successful = multi_manager.models[original].preload_instances(count)
        return {"message": f"成功預載入 {successful} 個模型實例", "successful": successful}

    @app.post("/model/cleanup")
    async def cleanup_models():
        return {"message": "清理請求已發送，將在下個清理周期執行"}

    @app.get("/test/echo")
    async def test_echo():
        return {"message": f"{device_config['name']} 伺服器運行正常"}

    # --------------------------------------------------------
    # OpenAI 相容接口（簡化 prompt，只取最後一條 user 訊息）
    # --------------------------------------------------------
    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def openai_chat(request: Request):
        try:
            data = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"無效的JSON格式: {str(e)}")

        # 解析模型 ID
        requested_model = data.get("model")
        if not requested_model:
            available = list(multi_manager.models.keys())
            if not available:
                raise HTTPException(status_code=503, detail="無可用模型")
            model_id = available[0]
            print(f"[警告] 未指定模型，使用預設 {model_id}")
        else:
            try:
                model_id = resolve_model_id(requested_model)
            except HTTPException as e:
                raise e

        if model_id not in multi_manager.models:
            raise HTTPException(status_code=400, detail=f"不支援的模型: {model_id}")

        # 取得最後一條 user 訊息作為 prompt
        messages = data.get("messages", [])
        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        if last_user_msg is None:
            raise HTTPException(status_code=400, detail="請求中沒有 user 角色的訊息")

        formatted_prompt = last_user_msg  # 直接使用純文字，不添加任何標籤

        result = multi_manager.get_model_instance(model_id, timeout=30)
        if result is None:
            raise HTTPException(status_code=503, detail="暫時沒有可用的模型實例，請稍後重試")

        request_id, model_instance = result
        tokenizer = model_instance["tokenizer"]

        try:
            tokenized = tokenizer.encode(formatted_prompt)
            if hasattr(tokenized.input_ids, 'shape') and len(tokenized.input_ids.shape) == 2:
                prompt_tokens = tokenized.input_ids.shape[1]
            else:
                prompt_tokens = int(tokenized.input_ids.get_shape()[1])
        except Exception as e:
            print(f"[錯誤] [{device_config['name']}:{model_id}#{request_id}] 計算token數失敗: {e}，改用文字長度估算")
            prompt_tokens = len(formatted_prompt) // 4

        max_len = multi_manager.models[model_id].extra_config.get("MAX_PROMPT_LEN", 16000)
        if prompt_tokens > max_len:
            multi_manager.release_model_instance(model_id, request_id)
            raise HTTPException(status_code=400, detail=f"提示詞過長 ({prompt_tokens} > {max_len})")

        chat_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        async def openai_gen():
            loop = asyncio.get_running_loop()
            streamer = AsyncStreamer(loop, request_id, model_instance.get("id", ""), model_id)

            future = thread_pool.submit(
                safe_generate_with_cleanup,
                model_instance, formatted_prompt, streamer,
                request_id, multi_manager, model_id
            )
            def _cb(fut):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[錯誤] [{device_config['name']}:{model_id}#{request_id}] 生成任務失敗: {e}")
            future.add_done_callback(_cb)

            # 發送第一個 chunk（角色訊息）
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': requested_model or model_id, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}}]})}\n\n"

            try:
                while True:
                    token = await asyncio.wait_for(streamer.queue.get(), timeout=300)
                    if token is None:
                        stats = streamer.get_stats()
                        usage_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": requested_model or model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": streamer.token_count,
                                "total_tokens": prompt_tokens + streamer.token_count,
                            },
                            "stats": {
                                "total_time": stats.get("total_time", 0),
                                "tokens_per_second": stats.get("tokens_per_second", 0),
                            }
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n"
                        break
                    yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': requested_model or model_id, 'choices': [{'index': 0, 'delta': {'content': token}}]})}\n\n"
            except asyncio.TimeoutError:
                print(f"[錯誤] [{device_config['name']}:{model_id}#{request_id}] 串流超時")
                raise HTTPException(status_code=408, detail="生成超時")
            except Exception as e:
                print(f"[錯誤] [{device_config['name']}:{model_id}#{request_id}] 串流錯誤: {e}")
                raise
            yield "data: [DONE]\n\n"

        return StreamingResponse(openai_gen(), media_type="text/event-stream")

    # --------------------------------------------------------
    # Ollama 相容接口（簡化 prompt，只取最後一條 user 訊息）
    # --------------------------------------------------------
    @app.post("/api/chat")
    @app.post("/api/generate")
    async def ollama_chat(request: Request):
        try:
            data = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"無效的JSON格式: {str(e)}")

        requested_model = data.get("model")
        if not requested_model:
            available = list(multi_manager.models.keys())
            if not available:
                raise HTTPException(status_code=503, detail="無可用模型")
            model_id = available[0]
            print(f"[警告] 未指定模型，使用預設 {model_id}")
        else:
            try:
                model_id = resolve_model_id(requested_model)
            except HTTPException as e:
                raise e

        if model_id not in multi_manager.models:
            raise HTTPException(status_code=400, detail=f"不支援的模型: {model_id}")

        # 優先使用 prompt 欄位，若無則從 messages 中取最後一條 user 內容
        prompt = data.get("prompt", "")
        if not prompt:
            messages = data.get("messages", [])
            last_user_msg = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
            if last_user_msg is None:
                raise HTTPException(status_code=400, detail="缺少 prompt 或 user 訊息")
            prompt = last_user_msg

        result = multi_manager.get_model_instance(model_id, timeout=30)
        if result is None:
            raise HTTPException(status_code=503, detail="暫時沒有可用的模型實例")

        request_id, model_instance = result
        tokenizer = model_instance["tokenizer"]

        try:
            tokenized = tokenizer.encode(prompt)
            if hasattr(tokenized.input_ids, 'shape') and len(tokenized.input_ids.shape) == 2:
                prompt_tokens = tokenized.input_ids.shape[1]
            else:
                prompt_tokens = int(tokenized.input_ids.get_shape()[1])
        except Exception:
            prompt_tokens = len(prompt) // 4

        max_len = multi_manager.models[model_id].extra_config.get("MAX_PROMPT_LEN", 16000)
        if prompt_tokens > max_len:
            multi_manager.release_model_instance(model_id, request_id)
            raise HTTPException(status_code=400, detail=f"提示詞過長 ({prompt_tokens} > {max_len})")

        start_time_ns = time.time_ns()

        async def ollama_gen():
            loop = asyncio.get_running_loop()
            streamer = AsyncStreamer(loop, request_id, model_instance.get("id", ""), model_id)

            future = thread_pool.submit(
                safe_generate_with_cleanup,
                model_instance, prompt, streamer,
                request_id, multi_manager, model_id
            )
            def _cb(fut):
                try:
                    fut.result()
                except Exception:
                    pass
            future.add_done_callback(_cb)

            try:
                while True:
                    token = await asyncio.wait_for(streamer.queue.get(), timeout=300)
                    if token is None:
                        end_ns = time.time_ns()
                        stats = streamer.get_stats()
                        total_duration_ns = end_ns - start_time_ns
                        yield json.dumps({
                            "model": requested_model or model_id,
                            "display_name": model_id,
                            "device": device_config["name"],
                            "done": True,
                            "prompt_eval_count": prompt_tokens,
                            "eval_count": streamer.token_count,
                            "total_duration": total_duration_ns,
                            "eval_duration": total_duration_ns,
                            "stats": {
                                "total_time_seconds": total_duration_ns / 1e9,
                                "tokens_per_second": stats.get("tokens_per_second", 0),
                            }
                        }) + "\n"
                        break
                    yield json.dumps({
                        "model": requested_model or model_id,
                        "display_name": model_id,
                        "device": device_config["name"],
                        "message": {"role": "assistant", "content": token},
                        "done": False
                    }) + "\n"
            except asyncio.TimeoutError:
                print(f"[錯誤] [{device_config['name']}:{model_id}#{request_id}] 串流超時")
                raise HTTPException(status_code=408, detail="生成超時")
            except Exception as e:
                print(f"[錯誤] [{device_config['name']}:{model_id}#{request_id}] 串流錯誤: {e}")
                raise

        return StreamingResponse(ollama_gen(), media_type="application/x-ndjson")

    return app