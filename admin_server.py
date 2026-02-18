# admin_server.py - 管理伺服器（提供網頁介面與 API，支援多模型）

import os
import json
import subprocess
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 導入模型倉儲模組
import model_registry

# 配置檔案路徑
CONFIG_FILE = Path("./device_config.json")
# 設備進程儲存
device_processes: Dict[str, subprocess.Popen] = {}

app = FastAPI(title="OpenVINO 模型管理伺服器 (多模型支援)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 完整 HTML 頁面內容（已更新，複選框改為垂直排列）
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenVINO 模型管理 (多模型)</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .section { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .section h2 { margin-top: 0; }
        label { display: inline-block; width: 80px; }
        select, input, button { padding: 5px; margin: 5px; }
        .status { font-weight: bold; }
        .running { color: green; }
        .stopped { color: red; }
        pre { background: #f4f4f4; padding: 10px; overflow: auto; max-height: 200px; }
        .flex { display: flex; gap: 10px; align-items: center; }
        .message { margin: 5px 0; padding: 5px; border-radius: 3px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .info { background: #d1ecf1; color: #0c5460; }
        .device-box { border: 1px solid #aaa; padding: 10px; margin-bottom: 10px; background: #f9f9f9; }
        /* 模型複選框樣式：每個獨佔一行，且內容不換行 */
        .model-checkbox {
            display: block;
            margin-bottom: 8px;
            white-space: nowrap;
        }
        .model-checkbox input {
            margin-right: 5px;
            vertical-align: middle;
        }
        .device-tag { background: #e0e0e0; border-radius: 3px; padding: 2px 5px; margin-left: 5px; font-size: 0.8em; }
        .model-container {
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>OpenVINO 模型管理 (多模型)</h1>

    <div class="section" id="pull-section">
        <h2>拉取新模型</h2>
        <div class="flex">
            <select id="model-select" style="min-width: 300px;">
                <option value="">-- 從列表選擇 --</option>
            </select>
            <span>或</span>
            <input type="text" id="custom-model-id" placeholder="手動輸入模型 ID (如 OpenVINO/phi-2-int4-ov)" size="40">
        </div>
        <div class="flex">
            <input type="text" id="revision" placeholder="分支/commit (選填)" size="20">
            <button onclick="pullModel()">開始拉取</button>
        </div>
        <div id="pull-message"></div>
    </div>

    <div class="section" id="models-section">
        <h2>本地已下載模型</h2>
        <ul id="model-list"></ul>
        <button onclick="refreshModels()">重新整理模型列表</button>
    </div>

    <div class="section" id="devices-section">
        <h2>設備管理</h2>
        <div id="devices"></div>
    </div>

    <div class="section" id="log-section">
        <h3>操作紀錄</h3>
        <pre id="log-content"></pre>
    </div>

    <script>
        const API_BASE = '';
        let localModels = [];

        async function fetchAPI(url, options = {}) {
            const resp = await fetch(API_BASE + url, options);
            if (!resp.ok) throw new Error(await resp.text());
            return resp.json();
        }

        async function loadAvailableModels() {
            try {
                const data = await fetchAPI('/api/available_models');
                const select = document.getElementById('model-select');
                select.innerHTML = '<option value="">-- 從列表選擇 --</option>';
                data.models.forEach(id => {
                    const option = document.createElement('option');
                    option.value = id;
                    option.textContent = id;
                    select.appendChild(option);
                });
                log('可拉取模型列表已更新');
            } catch (e) {
                log('取得可拉取模型列表失敗: ' + e);
            }
        }

        async function pullModel() {
            const select = document.getElementById('model-select');
            const customInput = document.getElementById('custom-model-id');
            const revision = document.getElementById('revision').value.trim();
            let modelId = select.value;
            if (!modelId && customInput.value.trim()) {
                modelId = customInput.value.trim();
            }
            if (!modelId) {
                showMessage('請選擇或輸入模型 ID', 'error');
                return;
            }
            const msgDiv = document.getElementById('pull-message');
            msgDiv.innerHTML = '<div class="message info">⏳ 開始下載，請稍候...</div>';
            try {
                const result = await fetchAPI('/api/pull_model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ model_id: modelId, revision: revision || undefined })
                });
                msgDiv.innerHTML = `<div class="message success">✅ ${result.message}</div>`;
                log(`模型 ${modelId} 下載成功`);
                refreshModels();
                refreshDevices();
            } catch (e) {
                msgDiv.innerHTML = `<div class="message error">❌ 下載失敗: ${e.message}</div>`;
                log(`模型 ${modelId} 下載失敗: ${e}`);
            }
        }

        function showMessage(text, type) {
            const msgDiv = document.getElementById('pull-message');
            msgDiv.innerHTML = `<div class="message ${type}">${text}</div>`;
        }

        async function refreshModels() {
            try {
                // 獲取本地模型列表
                const modelsData = await fetchAPI('/api/local_models');
                localModels = modelsData.models;
                // 獲取設備配置以知道每個模型分配給哪些設備
                const configData = await fetchAPI('/api/config');
                
                const list = document.getElementById('model-list');
                list.innerHTML = localModels.map(m => {
                    // 找出哪些設備使用了此模型
                    const assignedDevices = [];
                    for (const [dev, cfg] of Object.entries(configData)) {
                        if (cfg.models && cfg.models.some(selected => selected.id === m.id)) {
                            assignedDevices.push(dev.toUpperCase());
                        }
                    }
                    const deviceTags = assignedDevices.length > 0 
                        ? ` <span class="device-tag">使用於: ${assignedDevices.join(', ')}</span>` 
                        : '';
                    return `<li>${m.id}${deviceTags} (${(m.size/1e9).toFixed(2)} GB) - 下載於 ${new Date(m.downloaded_at*1000).toLocaleString()}</li>`;
                }).join('');
                log('本地模型列表已更新');
            } catch (e) {
                log('取得本地模型列表失敗: ' + e);
            }
        }

        // 儲存當前勾選狀態的物件
        let previousSelections = {};

        async function refreshDevices() {
            try {
                // 在重新繪製前，先保存當前所有勾選框的狀態
                document.querySelectorAll('.model-cb').forEach(cb => {
                    const device = cb.dataset.device;
                    const modelId = cb.dataset.modelId;
                    if (!previousSelections[device]) previousSelections[device] = {};
                    previousSelections[device][modelId] = cb.checked;
                });

                const data = await fetchAPI('/api/config');
                const devicesDiv = document.getElementById('devices');
                let html = '';
                for (const [name, cfg] of Object.entries(data)) {
                    const status = cfg.running ? '運行中' : '已停止';
                    const statusClass = cfg.running ? 'running' : 'stopped';
                    
                    // 建立模型複選框，每個獨立一行（使用 block 樣式）
                    const modelCheckboxesHtml = localModels.map(m => {
                        // 優先使用之前保存的狀態，若無則使用配置中的狀態
                        let checked = false;
                        if (previousSelections[name] && previousSelections[name][m.id] !== undefined) {
                            checked = previousSelections[name][m.id];
                        } else {
                            checked = cfg.models && cfg.models.some(selected => selected.id === m.id);
                        }
                        return `<label class="model-checkbox"><input type="checkbox" class="model-cb" data-device="${name}" data-model-id="${m.id}" ${checked ? 'checked' : ''}> ${m.id}</label>`;
                    }).join('');
                    
                    const modelContainer = `<div class="model-container">${modelCheckboxesHtml}</div>`;

                    html += `
                        <div class="device-box">
                            <h3>${name.toUpperCase()}</h3>
                            <p>端口: ${cfg.port}</p>
                            <p>狀態: <span class="status ${statusClass}">${status}</span></p>
                            <p>已啟用: <input type="checkbox" id="enabled-${name}" ${cfg.enabled ? 'checked' : ''}></p>
                            <div><strong>選擇模型：</strong>${modelContainer}</div>
                            <button onclick="applyConfig('${name}')">套用並重啟</button>
                        </div>
                    `;
                }
                devicesDiv.innerHTML = html;
                log('設備配置已更新');
            } catch (e) {
                log('取得設備配置失敗: ' + e);
            }
        }

        async function applyConfig(device) {
            // 收集該設備下所有勾選的模型
            const checkboxes = document.querySelectorAll(`.model-cb[data-device="${device}"]:checked`);
            const selectedModels = Array.from(checkboxes).map(cb => {
                const modelId = cb.dataset.modelId;
                const modelInfo = localModels.find(m => m.id === modelId);
                return {
                    id: modelId,
                    path: modelInfo ? modelInfo.path : '',
                    // 可加入其他預設值，如記憶體估計等（此處簡化，實際可讓用戶設定）
                    estimated_memory_gb: 6.0,
                    instance_memory_gb: 2.0,
                    max_concurrent: 5,
                    idle_timeout: 300
                };
            });

            const enabled = document.getElementById(`enabled-${device}`).checked;

            try {
                await fetchAPI('/api/update_config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ 
                        device, 
                        enabled,
                        models: selectedModels 
                    })
                });
                log(`已為 ${device} 更新配置，正在重啟...`);
                setTimeout(refreshDevices, 2000);
            } catch (e) {
                log('更新失敗: ' + e);
            }
        }

        function log(msg) {
            const pre = document.getElementById('log-content');
            pre.innerText = new Date().toLocaleTimeString() + ' ' + msg + '\\n' + pre.innerText;
        }

        loadAvailableModels();
        refreshModels();
        refreshDevices();
        // 將刷新間隔調整為 30 秒，減少重繪頻率，但仍保留狀態恢復邏輯
        setInterval(refreshDevices, 30000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def admin_page():
    return HTML_PAGE

# ---------- API 端點 ----------
@app.get("/api/available_models")
async def list_available_models():
    collections = model_registry.list_available_models()
    all_models = set()
    for models in collections.values():
        all_models.update(models)
    return {"models": sorted(all_models)}

@app.post("/api/pull_model")
async def pull_model(request: Request):
    data = await request.json()
    model_id = data.get("model_id")
    if not model_id:
        raise HTTPException(400, "缺少 model_id")
    revision = data.get("revision")
    success = model_registry.pull_model(model_id, revision)
    if success:
        return {"message": f"模型 {model_id} 下載成功"}
    else:
        raise HTTPException(500, f"下載模型 {model_id} 失敗")

@app.get("/api/local_models")
async def list_local_models():
    models = model_registry.list_local_models()
    return {"models": models}

@app.get("/api/config")
async def get_config():
    if not CONFIG_FILE.exists():
        default = {
            "gpu": {"enabled": True, "port": 11435, "models": []},
            "npu": {"enabled": True, "port": 11436, "models": []}
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default, f, indent=2)
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    for dev in config:
        config[dev]["running"] = dev in device_processes and device_processes[dev].poll() is None
    return config

@app.post("/api/update_config")
async def update_config(request: Request):
    data = await request.json()
    device = data.get("device")
    if device not in ["gpu", "npu"]:
        raise HTTPException(400, "無效的設備名稱")

    # 讀取現有配置
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # 更新該設備的啟用狀態和模型列表
    config[device]["enabled"] = data.get("enabled", config[device].get("enabled", True))
    config[device]["models"] = data.get("models", [])

    # 寫回配置
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

    # 重啟設備進程
    await restart_device(device, config[device])
    return {"status": "ok", "message": f"已更新 {device} 配置"}

async def restart_device(device: str, cfg: dict):
    # 終止舊進程
    if device in device_processes and device_processes[device].poll() is None:
        device_processes[device].terminate()
        try:
            device_processes[device].wait(timeout=10)
        except subprocess.TimeoutExpired:
            device_processes[device].kill()
        del device_processes[device]

    # 如果未啟用或沒有模型，則不啟動
    if not cfg.get("enabled") or not cfg.get("models"):
        return

    # 建立設備配置（傳遞給子進程）
    device_config = {
        "name": device.upper(),
        "device": "GPU.0" if device == "gpu" else "NPU",
        "port": cfg["port"],
        "models": cfg["models"]
    }

    cmd = [
        sys.executable, "-m", "uvicorn",
        "server_app:create_app",
        "--host", "127.0.0.1",
        "--port", str(cfg["port"]),
        "--factory",
    ]
    env = os.environ.copy()
    env["DEVICE_CONFIG_JSON"] = json.dumps(device_config)

    # 移除 log 檔案寫入，直接繼承父進程的 stdout/stderr
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=None,      # 繼承父進程的標準輸出
        stderr=None,      # 繼承父進程的標準錯誤
        text=True
    )
    device_processes[device] = proc

    # 等待短暫時間檢查是否啟動失敗
    time.sleep(2)
    if proc.poll() is not None:
        error_msg = f"設備 {device} 啟動失敗，退出碼: {proc.returncode}"
        del device_processes[device]
        raise HTTPException(status_code=500, detail=error_msg)

@app.on_event("startup")
async def startup_event():
    if not CONFIG_FILE.exists():
        default = {
            "gpu": {"enabled": True, "port": 11435, "models": []},
            "npu": {"enabled": True, "port": 11436, "models": []}
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default, f, indent=2)
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    for dev, cfg in config.items():
        if cfg.get("enabled") and cfg.get("models"):
            await restart_device(dev, cfg)

@app.on_event("shutdown")
async def shutdown_event():
    for dev, proc in device_processes.items():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=11437)