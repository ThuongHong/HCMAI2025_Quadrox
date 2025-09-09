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
MONGO_DB=quadrox
MONGO_USER=
MONGO_PASSWORD=
MONGO_URI=mongodb://localhost:27017/quadrox
GEMINI_API_KEY=<GEMINI_API_KEY>
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

<!-- 7. **Download model**


#### Model chính (Vintern - cần authentication):
```bash
# Đăng nhập HuggingFace CLI
huggingface-cli login --token <token>
# hoặc
hf login --token <token>

# Download Vintern model
hf download 5CD-AI/Vintern-1B-v3_5 --local-dir ./models/Vintern-1B-v3_5
```

#### Model multilingual embedding:
```bash
hf download sentence-transformers/clip-ViT-B-32-multilingual-v1 --local-dir ./models/clip-multilingual/clip-ViT-B-32-multilingual-v1
```

#### Models fallback (download tự động khi dùng):
- `Salesforce/blip-image-captioning-base`
- `microsoft/git-base-coco`

#### Precompute Captions (optional)

```bash
# Tạo caption trước cho tất cả keyframes
python tools/precompute_captions.py --images_glob "resources/keyframes/**/*.jpg"

# Với tham số custom
python tools/precompute_captions.py \
    --images_glob "resources/keyframes/**/*.jpg" \
    --style dense \
    --max_new_tokens 64 \
    --max_workers 4 \
    --batch_size 20
``` -->

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

## Temporal Search (beta)

We added opt-in Temporal Search to improve KIS/TRAKE metrics without changing existing behavior.

- Auto mode: Expands a window ±5→20s around a pivot keyframe using Adaptive Bidirectional Temporal Search (ABTS) and clusters keyframes by time (default gap 10s).
- Interactive mode: Pick a pivot result and choose ±Δ seconds to browse neighboring keyframes chronologically.

Backend:
- New endpoint `POST /keyframe/temporal/enrich`.
- Request fields: `mode` (auto|interactive), `pivot_video_id` (e.g., L01_V001) or (`pivot_group_num`,`pivot_video_num`), and either `pivot_n` or (`pivot_frame_idx` & `pivot_pts_time`), optional `delta` for interactive.
- Response includes `pivot`, time `window`, and `clusters` with representative keyframe per cluster.

Frontend:
- In Streamlit, open “Temporal Search (beta)” under results. Toggle on, choose mode, and preview clusters as horizontal strips. Thumbnails are limited to keyframes; no video decoding required.

Implementation notes:
- Uses `resources/map-keyframes/<video_id>.csv` with columns `n, pts_time, fps, frame_idx`.
- Clustering groups by contiguous time with default 10s gap.
- When similarity is unavailable per keyframe, temporal decay by distance to pivot provides stable expansion and ordering.
