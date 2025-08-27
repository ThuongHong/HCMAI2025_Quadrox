# HCMAI2025_Baseline


### 🔧 Local Development

1. Install uv and setup env
```bash
pip install uv
uv init --python=3.10
uv add aiofiles beanie dotenv fastapi[standard] httpx ipykernel motor nicegui numpy open-clip-torch pydantic-settings pymilvus streamlit torch typing-extensions usearch uvicorn
```

2. Activate .venv
```bash
source/SCripts/activate
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