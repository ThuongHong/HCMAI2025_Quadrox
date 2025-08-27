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

### 🔧 Local Development

1. Install uv and setup env
```bash
uv sync
```

2. Activate .venv
```bash
.venv/Scripts/activate
```

3. Run docker compose
```bash
docker compose up -d
```

4. Data Migration 
```bash
python migration/npy_embedding_migration.py --folder_path <folder_path>
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