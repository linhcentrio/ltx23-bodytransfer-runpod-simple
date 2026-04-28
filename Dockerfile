FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    WORKSPACE_DIR=/workspace \
    COMFY_ROOT=/ComfyUI \
    INPUT_DIR=/workspace/input \
    OUTPUT_DIR=/workspace/output \
    MODELS_DIR=/ComfyUI/models \
    AUX_ANNOTATOR_DIR=/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts \
    PRELOAD_MODELS_ON_STARTUP=false \
    MODEL_QUALITY=balanced_q3 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget aria2 ffmpeg libsndfile1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    build-essential pkg-config cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      runpod \
      requests \
      websocket-client \
      minio \
      pillow \
      numpy \
      opencv-python-headless \
      soundfile \
      moviepy \
      av \
      scipy \
      einops \
      diffusers \
      accelerate \
      transformers \
      torchsde \
      safetensors \
      gguf \
      huggingface_hub[hf_transfer] \
      onnx \
      onnxruntime-gpu && \
    (pip install --no-cache-dir xformers || true) && \
    (pip install --no-cache-dir triton sageattention || true)

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /ComfyUI && \
    pip install --no-cache-dir -r /ComfyUI/requirements.txt

RUN mkdir -p /ComfyUI/custom_nodes && \
    cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    for d in ComfyUI-KJNodes ComfyUI-GGUF ComfyUI-VideoHelperSuite comfyui_controlnet_aux ComfyUI-LTXVideo; do \
      if [ -f "/ComfyUI/custom_nodes/$d/requirements.txt" ]; then pip install --no-cache-dir -r "/ComfyUI/custom_nodes/$d/requirements.txt" || true; fi; \
    done && \
    rm -rf /ComfyUI/custom_nodes/ComfyUI-TeaCache || true

RUN mkdir -p /ComfyUI/models/unet/LTX2 \
             /ComfyUI/models/clip \
             /ComfyUI/models/vae/LTX-2.3 \
             /ComfyUI/models/latent_upscale_models \
             /ComfyUI/models/loras/LTX-2.3 \
             /ComfyUI/models/checkpoints/SDPose \
             /ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d /ComfyUI/models/unet/LTX2 -o ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/distilled-1.1/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d /ComfyUI/models/clip -o gemma-3-12b-it-Q2_K.gguf https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q2_K.gguf && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d /ComfyUI/models/clip -o ltx-2.3_text_projection_bf16.safetensors https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d /ComfyUI/models/vae/LTX-2.3 -o LTX23_video_vae_bf16.safetensors https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d /ComfyUI/models/vae/LTX-2.3 -o LTX23_audio_vae_bf16.safetensors https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d /ComfyUI/models/vae/LTX-2.3 -o taeltx2_3.safetensors https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d /ComfyUI/models/latent_upscale_models -o ltx-2.3-spatial-upscaler-x2-1.0.safetensors https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d /ComfyUI/models/loras/LTX-2.3 -o ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d /ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose -o yolox_l.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d /ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose -o dw-ll_ucoco_384.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d /ComfyUI/models/checkpoints/SDPose -o sdpose_wholebody_fp16.safetensors https://huggingface.co/Comfy-Org/SDPose/resolve/main/checkpoints/sdpose_wholebody_fp16.safetensors

COPY handler.py /workspace/handler.py
RUN chmod +x /workspace/handler.py && mkdir -p /workspace/input /workspace/output /workspace/tmp

CMD ["python", "-u", "/workspace/handler.py"]
