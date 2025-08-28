# HCMAI2025_Baseline

A FastAPI-based AI application powered by Milvus for vector search, MongoDB for metadata storage, and MinIO for object storage.

## ✨ Features

### 🔍 Search Capabilities

- **Text Search**: Search keyframes using natural language queries with semantic similarity
- **Image Search**: Upload images to find visually similar keyframes
- **Metadata Filtering**: Filter results by video metadata (author, keywords, length, title, description, date)
- **🎯 Object Detection Filtering**: NEW! Filter keyframes by detected objects with two modes:
  - **Any mode**: Find keyframes containing at least one of the specified objects
  - **All mode**: Find keyframes containing all specified objects
- **Group/Video Filtering**: Include or exclude specific groups and videos
- **Combined Filtering**: Combine metadata and object filters for precise results

### 🎯 Object Detection Filtering

Filter keyframes by objects detected in the images:

- **Supported Objects**: Cars, people, buildings, skyscrapers, trees, vehicles, and many more
- **Two Filter Modes**:
  - `any`: Keyframes containing at least one of the specified objects
  - `all`: Keyframes containing all of the specified objects
- **Case-insensitive matching**: "Car", "car", and "CAR" are treated the same
- **Smart deduplication**: Duplicate objects in the filter list are automatically removed
- **Performance optimized**: Uses MongoDB indexes for fast object filtering
- **Combined with metadata**: Use both object and metadata filters simultaneously

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
python migration/npy_embedding_migration.py --folder_path resources/embeddings
python migration/keyframe_migration.py --file_path resources/keyframes/id2index.json
python migration/metadata_migration.py --folder_path resources/metadata
python migration/objects_migration.py --folder_path resources/objects
```

5. Run the application

Open 2 tabs

5.1. Run the FastAPI application

```bash
cd app
python main.py
```

5.2. Run the Streamlit application

```bash
cd gui
streamlit run main.py
```