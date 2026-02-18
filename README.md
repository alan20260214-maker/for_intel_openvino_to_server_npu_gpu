
# OpenVINO 多模型管理伺服器 (for Intel NPU/GPU)

這是一個專為 **Intel 架構 (GPU/NPU)** 設計的 AI 模型管理與服務系統。它能讓你在同一台機器上，同時為多個不同的大型語言模型（LLM）提供高效率的服務，並智慧管理 GPU 與 NPU 的記憶體資源。

### ✨ 主要特性
*   **多設備支援**：可同時管理 **GPU** 與 **NPU** 上的模型服務進程。
*   **多模型併存**：同一個設備（如 GPU）可動態載入多個不同的模型，並在請求之間共享記憶體。
*   **記憶體感知管理**：每個模型都有獨立的實例池，系統會根據實際記憶體使用情況，自動調整最大併發數，防止記憶體不足。
*   **友善的管理介面**：透過網頁 (`http://127.0.0.1:11437`) 即可輕鬆下載新模型（Hugging Face）、指派模型到各設備、啟動/停止服務。
*   **OpenAI 與 Ollama 相容 API**：每個設備的服務埠（GPU:11435, NPU:11436）提供與 OpenAI `/v1/chat/completions` 及 Ollama `/api/chat` 相容的 API，方便整合現有工具。
*   **智慧實例池**：模型實例會在被釋放後保留一段時間（閒置逾時），以減少重複載入的延遲。

## 🚀 快速開始

### 前置需求
*   **作業系統**：Windows 或 Linux（需支援 Intel GPU/NPU 驅動）
*   **Python**：3.9 或更高版本
*   **Intel OpenVINO**：需安裝 `openvino-genai` 套件（請參考 [OpenVINO 官方安裝指南](https://docs.openvino.ai/)）
*   **硬體**：Intel 整合顯卡、獨立顯卡（GPU）或 NPU（如 Intel Core Ultra 處理器）

### 安裝步驟
1.  **克隆專案**
    
    git clone https://github.com/alan20260214-maker/for_intel_openvino_to_server_npu_gpu.git
    cd for_intel_openvino_to_server_npu_gpu
    
2.  **建立虛擬環境（建議）**
    
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    
3.  **安裝依賴套件**
    專案依賴主要為 `fastapi`, `uvicorn`, `huggingface_hub`, `psutil`, `humanize` 以及 `openvino-genai`。
    
    pip install fastapi uvicorn huggingface_hub psutil humanize
    # 請根據你的環境安裝 openvino-genai，例如：
    # pip install openvino-genai
    
4.  **啟動管理伺服器**
    
    python main.py
    
5.  **開啟管理介面**
    使用瀏覽器開啟 [http://127.0.0.1:11437](http://127.0.0.1:11437) 即可開始使用。

## 📖 使用指南

### 1. 拉取模型
在網頁介面的「拉取新模型」區塊，你可以從預設的 OpenVINO 模型集合中選擇，或手動輸入 Hugging Face 上的模型 ID（例如 `OpenVINO/Phi-3-medium-4k-instruct-int4-ov`）。支援指定分支或 commit。

### 2. 指派模型到設備
下載完成後，模型會出現在「本地已下載模型」列表中。接著，在「設備管理」區塊，你可以為 GPU 和 NPU 分別勾選想要啟用的模型。

### 3. 套用設定
點擊設備區塊的「套用並重啟」按鈕，系統會：
*   更新設定檔 `device_config.json`。
*   重新啟動對應的模型服務進程（GPU 服務在埠 `11435`，NPU 服務在埠 `11436`）。
*   新設定會立即生效。

### 4. 使用 API 與模型互動
每個設備都是一個獨立的 FastAPI 伺服器，提供相容的 API 端點。

**範例：呼叫 GPU 上的 Phi-3 模型 (OpenAI 相容)**

curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpu-OpenVINO/Phi-3-medium-4k-instruct-int4-ov",
    "messages": [{"role": "user", "content": "什麼是 OpenVINO？"}]
  }'

**注意**：API 中使用的模型 ID 格式為 `{設備前綴}-{原始模型ID}`，例如 `gpu-OpenVINO/Phi-3-medium-4k-instruct-int4-ov`。這是為了讓客戶端明確指定要使用的設備。

## ⚙️ 設定檔說明 (`device_config.json`)
管理伺服器會自動建立並維護此檔案。你也可以手動編輯它。
*   `gpu` / `npu`：設備設定。
    *   `enabled`：是否啟用此設備服務。
    *   `port`：服務監聽的埠號。
    *   `models`：為該設備啟用的模型列表，每個模型包含 `id`（模型 ID）、`path`（本地路徑）、`estimated_memory_gb`（預估權重記憶體）、`instance_memory_gb`（每個實例額外記憶體）等參數。

## 🗂️ 專案結構

├── admin_server.py      # 管理後台伺服器 (FastAPI + HTML)
├── main.py              # 啟動管理伺服器的入口
├── model_manager.py     # 核心：記憶體感知的多模型管理器
├── model_registry.py    # 模型下載、列表管理（Hugging Face 整合）
├── server_app.py        # 實際服務模型 API 的應用程式（供子進程使用）
├── device_config.json   # 設備與模型設定檔（自動產生）
├── models/              # 下載的模型存放目錄（自動建立）
├── cache_*/             # OpenVINO 快取目錄（自動建立）
├── THIRD_PARTY_NOTICES.txt # 第三方套件授權聲明
└── LICENSE              # MIT 授權檔案


## 🔗 第三方開源元件授權
本專案使用了許多優秀的開源函式庫，其授權資訊請詳閱 [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt) 檔案。

## 📜 授權條款
本專案程式碼採用 **MIT 授權**，詳情請參閱 [`LICENSE`](LICENSE) 檔案。
**重要提醒**：本專案僅為管理工具。當你透過本專案下載或使用 Hugging Face 或其他來源的 AI 模型時，**你必須自行遵守該模型本身所適用的授權條款**。部分模型可能僅限研究用途，或用於商業用途時需另行取得授權。

## 🤝 貢獻
歡迎任何形式的貢獻！如果你有想法或改進，請提交 Issue 或 Pull Request。

