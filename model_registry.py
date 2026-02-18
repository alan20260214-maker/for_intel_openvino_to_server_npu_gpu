# model_registry.py - 模型倉儲管理（支援重試、Token 驗證、斷點續傳）

import os
import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional
import requests
from huggingface_hub import snapshot_download, HfFileSystem
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

MODELS_BASE_DIR = Path("./models")
MODELS_BASE_DIR.mkdir(exist_ok=True)

HF_COLLECTION_API = "https://huggingface.co/api/collections/OpenVINO/{collection}"

COLLECTIONS = {
    "llm": "llm",
    "npu": "llms-optimized-for-npu"
}

def fetch_collection_models(collection: str) -> List[str]:
    """從 Hugging Face 集合中取得模型 ID 列表"""
    url = HF_COLLECTION_API.format(collection=collection)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        models = []
        for item in data.get('items', []):
            if item.get('type') == 'model':
                model_id = item.get('id')
                if model_id:
                    models.append(model_id)
        return models
    except Exception as e:
        print(f"❌ 獲取集合 {collection} 失敗: {e}")
        return []

def list_available_models() -> Dict[str, List[str]]:
    """列出所有可拉取的模型（依集合分類）"""
    result = {}
    for key, collection in COLLECTIONS.items():
        models = fetch_collection_models(collection)
        result[key] = models
    return result

def list_local_models() -> List[Dict]:
    """列出本地已下載的模型及其 metadata"""
    local_models = []
    for model_dir in MODELS_BASE_DIR.iterdir():
        if model_dir.is_dir():
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                # 若無 metadata 則自動產生（向後相容）
                metadata = {
                    "id": model_dir.name.replace('_', '/'),
                    "path": str(model_dir),
                    "size": sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()),
                    "downloaded_at": model_dir.stat().st_ctime
                }
            local_models.append(metadata)
    return local_models

def pull_model(model_id: str, revision: Optional[str] = None, max_retries: int = 3) -> bool:
    """
    下載 Hugging Face 模型到本地，支援重試與 Token 驗證。
    已移除 ignore_patterns，確保完整下載所有檔案。
    """
    safe_dir_name = model_id.replace('/', '_')
    target_dir = MODELS_BASE_DIR / safe_dir_name

    # 若目標目錄已存在且包含完整的 metadata.json，可跳過下載（依需求可改為強制重新下載）
    if target_dir.exists() and (target_dir / "metadata.json").exists():
        print(f"⚠️ 模型 {model_id} 已存在於 {target_dir}，跳過下載")
        return True

    # 讀取環境變數中的 Hugging Face Token（用於存取 gated models）
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("🔑 使用環境變數 HF_TOKEN 進行身份驗證")

    # 重試邏輯
    for attempt in range(1, max_retries + 1):
        try:
            print(f"⏳ 正在下載模型 {model_id} 到 {target_dir} ... (嘗試 {attempt}/{max_retries})")

            snapshot_download(
                repo_id=model_id,
                local_dir=target_dir,
                revision=revision,
                local_dir_use_symlinks=False,   # 強制複製檔案而非符號連結
                token=hf_token,                  # 傳入 token（若為 None 則不帶身份驗證）
                # resume_download=True,           # 啟用斷點續傳（huggingface_hub 0.20+ 支援）
            )

            # 下載完成後寫入 metadata
            metadata = {
                "id": model_id,
                "path": str(target_dir),
                "downloaded_at": time.time(),
                "revision": revision,
                "size": sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file())
            }
            with open(target_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ 模型 {model_id} 下載完成")
            return True

        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            # 倉庫或版本不存在，無需重試
            print(f"❌ 模型 {model_id} 不存在或 revision 錯誤: {e}")
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            return False

        except Exception as e:
            print(f"⚠️ 下載嘗試 {attempt} 失敗: {e}")
            if attempt == max_retries:
                print(f"❌ 下載模型 {model_id} 最終失敗，已達最大重試次數")
                # 清理可能不完整的目錄
                if target_dir.exists():
                    shutil.rmtree(target_dir, ignore_errors=True)
                return False
            else:
                # 等待一段時間後重試（指數退避）
                wait_time = 2 ** attempt
                print(f"⏱️ 等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
    return False

def remove_model(model_id: str) -> bool:
    """從本地刪除指定模型"""
    possible_names = [
        model_id.replace('/', '_'),
        model_id
    ]
    for name in possible_names:
        target_dir = MODELS_BASE_DIR / name
        if target_dir.exists():
            try:
                shutil.rmtree(target_dir)
                print(f"🗑️ 模型 {model_id} 已移除")
                return True
            except Exception as e:
                print(f"❌ 移除模型 {model_id} 失敗: {e}")
                return False
    print(f"⚠️ 模型 {model_id} 不存在")
    return False