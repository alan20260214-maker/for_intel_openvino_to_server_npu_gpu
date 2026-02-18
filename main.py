# main.py - 啟動管理伺服器（不再直接啟動 GPU/NPU 進程）

import os
import sys

if __name__ == "__main__":
    print("🚀 啟動 OpenVINO 模型管理伺服器 (端口 11437)")
    print("📡 請開啟瀏覽器訪問 http://127.0.0.1:11437")
    # 導入管理伺服器
    from admin_server import app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=11437)