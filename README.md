# HCMAI2025_Baseline

A FastAPI-based AI application powered by Milvus for vector search, MongoDB for metadata storage, and MinIO for object storage.

## 🧑‍💻 Getting Started

### Prerequisites
- Docker
- Docker Compose
- Python 3.10
- uv

### 🔧 Local Development
1. Clone the repo and start all services:
```bash
git clone https://github.com/ThuongHong/HCMAI2025_Quadrox.git
```

2. uv sync
```bash
uv sync
```

3. Activate .venv
```bash
.venv/Scripts/activate
```
4. Run docker compose
```bash
docker compose up -d
```

4. Data Migration 
```bash
python migration/embedding_migration.py --file_path <emnedding.pt file>
python migration/keyframe_migration.py --file_path <id2index.json file path>
```

5. Run the application

Open 2 tabs

5.1. Run the FastAPI application
```bash
cd gui
streamlit run main.py
```

5.1. Run the Streamlit application
```bash
cd app
python main.py
```