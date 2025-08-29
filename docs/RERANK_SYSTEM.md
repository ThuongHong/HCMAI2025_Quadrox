# 🎯 Multi-Stage Reranking System

Hệ thống Multi-Stage Reranking cho Textual KIS (Keyframe Image Search) với các tùy chọn UI/API hoàn chỉnh.

## 📋 Tổng quan

Hệ thống reranking multi-stage được thiết kế để cải thiện chất lượng kết quả tìm kiếm thông qua 3 giai đoạn:

1. **SuperGlobal Rerank** - Reranking nhanh sử dụng global features
2. **Caption Rerank** - Reranking sử dụng Vietnamese image captions
3. **LLM Rerank** - Reranking chất lượng cao sử dụng Large Language Models

## 🏗️ Kiến trúc

```
Query → Initial Search → SuperGlobal → Caption → LLM → Final Results
                         (top_T1)     (top_T2)   (top_T3)
```

### Cấu trúc thư mục

```
app/retrieval/rerank/
├── __init__.py
├── options.py          # RerankOptions class với validation
├── pipeline.py         # RerankPipeline orchestrator
├── superglobal.py      # SuperGlobal reranker
├── captioning.py       # Caption reranker (Vietnamese)
├── llm_ranker.py       # LLM reranker
└── ensemble.py         # Ensemble scoring
```

## ⚙️ Cấu hình

### Settings (app/core/settings.py)

```python
class RerankSettings:
    # SuperGlobal settings
    rerank_superglobal_enabled: bool = True
    rerank_superglobal_weight: float = 0.4
    rerank_superglobal_top_t: int = 150

    # Caption settings
    rerank_caption_enabled: bool = True
    rerank_caption_weight: float = 0.4
    rerank_caption_top_t: int = 50
    rerank_caption_timeout: int = 30

    # LLM settings
    rerank_llm_enabled: bool = False
    rerank_llm_weight: float = 0.2
    rerank_llm_top_t: int = 20
    rerank_llm_timeout: int = 60

    # Advanced settings
    rerank_cache_enabled: bool = True
    rerank_fallback_enabled: bool = True
    rerank_final_top_k: int = 0
```

### Precedence Rules

```
Request Parameters > Environment Variables > Config Settings
```

## 🔗 API Endpoints

### 1. Basic Search với Rerank

```http
POST /api/v1/keyframe/search
Content-Type: application/json

{
  "query": "người đàn ông đang đi bộ",
  "top_k": 20,
  "score_threshold": 0.2,
  "rerank_superglobal_enabled": true,
  "rerank_superglobal_weight": 0.4,
  "rerank_caption_enabled": true,
  "rerank_caption_weight": 0.4
}
```

### 2. Image Search với Rerank

```http
POST /api/v1/keyframe/search/image
Content-Type: multipart/form-data

file: [image file]
top_k: 20
rerank_superglobal_enabled: true
rerank_caption_enabled: true
```

### 3. Advanced Search (Full Features)

```http
POST /api/v1/keyframe/search/advanced
Content-Type: application/json

{
  "query": "xe hơi màu đỏ",
  "top_k": 20,
  "rerank_superglobal_enabled": true,
  "rerank_superglobal_weight": 0.4,
  "rerank_superglobal_top_t": 150,
  "rerank_caption_enabled": true,
  "rerank_caption_weight": 0.4,
  "rerank_caption_top_t": 50,
  "rerank_llm_enabled": false,
  "metadata_filter": {
    "keywords": ["tin tuc", "HTV"]
  },
  "object_filter": {
    "objects": ["car", "person"],
    "mode": "any"
  }
}
```

## 🖥️ GUI Usage

### Kích hoạt Reranking

1. Mở section **"⚡ Rerank Options"**
2. Check **"Enable Multi-Stage Reranking"**
3. Cấu hình các stages cần thiết

### Các nút Search

- **🚀 Text Search** - Tìm kiếm chuẩn với rerank tùy chọn
- **🖼️ Image Search** - Tìm kiếm visual với rerank
- **⚡ Advanced Search** - Tìm kiếm full-featured sử dụng endpoint `/search/advanced`

### Cấu hình Stages

#### SuperGlobal Stage

- **Weight**: 0.0 - 1.0 (default: 0.4)
- **Top_T**: 50 - 500 (default: 150)
- **Mô tả**: Reranking nhanh sử dụng global features

#### Caption Stage

- **Weight**: 0.0 - 1.0 (default: 0.4)
- **Top_T**: 10 - 100 (default: 50)
- **Timeout**: 10 - 120s (default: 30s)
- **Mô tả**: Reranking sử dụng Vietnamese captions

#### LLM Stage

- **Weight**: 0.0 - 1.0 (default: 0.2)
- **Top_T**: 5 - 50 (default: 20)
- **Timeout**: 30 - 300s (default: 60s)
- **Mô tả**: Reranking chất lượng cao với LLM

## 🧪 Testing

### CLI Demo Tool

```bash
python tools/demo_kis_rerank.py --help
```

### GUI Demo Instructions

```bash
python tools/demo_gui_rerank.py
```

### Test Scenarios

#### Scenario 1: SuperGlobal Only

```bash
python tools/demo_kis_rerank.py \
  --query "người đàn ông" \
  --rerank-superglobal \
  --superglobal-weight 0.4
```

#### Scenario 2: SuperGlobal + Caption

```bash
python tools/demo_kis_rerank.py \
  --query "xe hơi đỏ" \
  --rerank-superglobal \
  --rerank-caption \
  --caption-top-t 25
```

#### Scenario 3: Full Pipeline

```bash
python tools/demo_kis_rerank.py \
  --query "con mèo" \
  --rerank-superglobal \
  --rerank-caption \
  --rerank-llm \
  --llm-top-t 10
```

## 🔧 Implementation Details

### Validation Logic

```python
# Tự động clamp values
superglobal_top_t = max(50, min(500, superglobal_top_t))
caption_weight = max(0.0, min(1.0, caption_weight))

# Kiểm tra dependencies
if caption_enabled and superglobal_top_t < caption_top_t:
    # Auto-adjust or raise error
```

### Error Handling

- **Graceful Degradation**: Fallback to simpler methods on failure
- **Timeout Protection**: Configurable timeouts for each stage
- **Cache Integration**: File-based caching for expensive operations

### Caching Strategy

```python
# Cache keys
cache_key = f"caption_{query_hash}_{model_hash}"
cache_file = cache_dir / f"{cache_key}.json"

# Cache structure
{
  "query": "original_query",
  "results": [...],
  "timestamp": "2024-01-15T10:30:00",
  "model_version": "v1.0"
}
```

## 🚀 Deployment

### Environment Variables

```bash
# SuperGlobal
RERANK_SUPERGLOBAL_ENABLED=true
RERANK_SUPERGLOBAL_WEIGHT=0.4

# Caption
RERANK_CAPTION_ENABLED=true
RERANK_CAPTION_WEIGHT=0.4
RERANK_CAPTION_TIMEOUT=30

# LLM
RERANK_LLM_ENABLED=false
RERANK_LLM_WEIGHT=0.2
RERANK_LLM_TIMEOUT=60

# Advanced
RERANK_CACHE_ENABLED=true
RERANK_FALLBACK_ENABLED=true
```

### Docker Compose

```yaml
services:
  app:
    environment:
      - RERANK_SUPERGLOBAL_ENABLED=true
      - RERANK_CAPTION_ENABLED=true
      - RERANK_LLM_ENABLED=false
    volumes:
      - ./cache:/app/cache # For rerank caching
```

## 📊 Performance

### Benchmarks (ước tính)

- **SuperGlobal**: ~50ms per 150 candidates
- **Caption**: ~2-5s per 50 candidates
- **LLM**: ~10-30s per 20 candidates
- **Total Pipeline**: ~15-35s for full rerank

### Optimization Tips

1. **Tune top_T values** - Smaller = faster
2. **Enable caching** - 10x speedup for repeated queries
3. **Adjust timeouts** - Balance quality vs speed
4. **Use selective stages** - Not all queries need LLM
5. **Parallel processing** - Future enhancement

## 🔍 Troubleshooting

### Common Issues

#### Rerank not applying

```bash
# Check settings
curl http://localhost:8000/api/v1/system/settings | jq .rerank

# Verify request
curl -X POST http://localhost:8000/api/v1/keyframe/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "rerank_superglobal_enabled": true}'
```

#### Performance issues

```bash
# Check cache directory
ls -la cache/rerank/

# Monitor logs
tail -f logs/main.log | grep -i rerank
```

#### Model loading errors

```bash
# Caption model check
python -c "from app.retrieval.rerank.captioning import CaptionRanker; r = CaptionRanker()"

# LLM model check
python -c "from app.retrieval.rerank.llm_ranker import LLMRanker; r = LLMRanker()"
```

## 🎯 Future Enhancements

1. **Real Model Integration**

   - Replace mock implementations with real models
   - Add model versioning and A/B testing

2. **Advanced Features**

   - Multi-modal fusion (text + image + audio)
   - Personalization based on user history
   - Dynamic weight adjustment

3. **Performance Optimizations**

   - GPU acceleration for models
   - Distributed processing
   - Smart caching strategies

4. **UI/UX Improvements**
   - Real-time rerank preview
   - Performance metrics display
   - Rerank explanation/interpretability

## 📄 License

MIT License - See LICENSE file for details.

---

_Được phát triển cho HCMAI2025_Quadrox project_
