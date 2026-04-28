# LTX23 BodyTransfer RunPod Simple

Bản refactor tối giản cho RunPod Serverless.

## Files
- `Dockerfile`
- `handler.py`
- `.env.example`

## Input schema
```json
{
  "input": {
    "source_video_url": "https://.../source.mp4",
    "control_video_url": "https://.../control.mp4",
    "prompt": "A person dancing naturally"
  }
}
```

### Optional long-video controls
```json
{
  "input": {
    "fps": 12,
    "max_frames": 721,
    "long_video_mode": true,
    "chunk": 8,
    "decode_tile": 256,
    "temporal_size": 512
  }
}
```

## Required env
```env
MINIO_ENDPOINT=media.aiclip.ai
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
MINIO_BUCKET=video
MINIO_SECURE=false
MINIO_PUBLIC_BASE_URL=http://media.aiclip.ai
MODEL_QUALITY=balanced_q3
DEFAULT_FPS=12
LONG_VIDEO_MODE=true
LONG_VIDEO_MAX_FRAMES=721
DEFAULT_CHUNK=4
LONG_VIDEO_CHUNK=8
DEFAULT_DECODE_TILE=384
LONG_VIDEO_DECODE_TILE=256
DEFAULT_TEMPORAL_SIZE=1024
LONG_VIDEO_TEMPORAL_SIZE=512
```

## Build image
```bash
git init
git add .
git commit -m "refactor: runpod simple bodytransfer"

# nếu dùng GitHub repo mới
# git remote add origin <repo-url>
# git push -u origin main
```

## RunPod deploy notes
- Base image đã bake sẵn ComfyUI + nodes + core models.
- Mặc định không cần `preset`; output size và length sẽ tự bám theo video control/ref.
- Chế độ dài mặc định hạ FPS/chunk/decode tile để cố gắng chạy được clip khoảng 50-60s tùy GPU/VRAM.
- Nếu endpoint cũ đang lỗi startup, tạo endpoint mới để tránh state cũ.
- Nên set min workers = 1 khi smoke test.

## Smoke test
```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <RUNPOD_API_KEY>' \
  -d '{
    "input": {
      "source_video_url": "https://.../source.mp4",
      "control_video_url": "https://.../control.mp4",
      "prompt": "A person dancing naturally",
      "fps": 12,
      "max_frames": 721,
      "long_video_mode": true
    }
  }'
```

## Important
- Worker cần truy cập public URL của input video.
- Nếu startup vẫn fail, xem container stdout/stderr trước, không chỉ worker dashboard.
- `handler.py` hiện là bản self-contained để deploy nhanh; có thể tách nhỏ lại sau khi endpoint chạy ổn định.
