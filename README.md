# HCMAI2025_Baseline

A FastAPI-based AI application powered by Milvus for vector search, MongoDB for metadata storage, and MinIO for object storage.

## 🧑‍💻 Getting Started

### Prerequisites

- Docker
- Docker Compose
- Python 3.10+
- uv (Python package manager)

### 🔧 Local Development

1. **Clone the repository**

```bash
git clone https://github.com/ThuongHong/HCMAI2025_Quadrox.git
cd HCMAI2025_Quadrox
```

2. **Install dependencies and setup environment**

```bash
uv sync
```

3. **Setup environment variables**

Create a `.env` file in the root directory with the following variables:

```env
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=<MONGO_DB>
MONGO_USER=<your_username>
MONGO_PASSWORD=<your_password>
MONGO_URI=<MONGO_URI>
GEMINI_API_KEY=<your_gemini_api_key_here>
```

4. **Activate virtual environment**

```bash
# Windows
.venv/Scripts/activate

# Linux/Mac
source .venv/bin/activate
```

5. **Start services with Docker**

```bash
docker compose up -d
```

6. **Run data migration scripts**

Execute the following migration scripts in order:

```bash
# Migrate embeddings
python migration/npy_embedding_migration.py --folder_path resources/embeddings

# Migrate keyframe data
python migration/keyframe_migration.py --file_path resources/keyframes/id2index.json

# Migrate video metadata
python migration/metadata_migration.py --folder_path resources/metadata

# Migrate object detection data
python migration/objects_migration.py --folder_path resources/objects
```

7. **Run the applications**

Open two terminal tabs:

**Tab 1 - FastAPI Backend:**
```bash
cd app
python main.py
```

**Tab 2 - Streamlit Frontend:**
```bash
cd gui
streamlit run main.py
```

## 📝 License

This project is part of the HCMAI2025 competition.