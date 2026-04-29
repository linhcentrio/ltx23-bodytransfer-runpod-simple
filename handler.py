import asyncio
import gc
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import quote

import requests
import runpod
import torch
from minio import Minio

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger('ltx23-bodytransfer')

WORKSPACE_DIR = Path(os.getenv('WORKSPACE_DIR', '/workspace'))
COMFY_ROOT = Path(os.getenv('COMFY_ROOT', '/ComfyUI'))
INPUT_DIR = Path(os.getenv('INPUT_DIR', str(WORKSPACE_DIR / 'input')))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', str(WORKSPACE_DIR / 'output')))
MODELS_DIR = Path(os.getenv('MODELS_DIR', str(COMFY_ROOT / 'models')))
AUX_ANNOTATOR_DIR = Path(os.getenv('AUX_ANNOTATOR_DIR', str(COMFY_ROOT / 'custom_nodes' / 'comfyui_controlnet_aux' / 'ckpts')))

MODEL_QUALITY = os.getenv('MODEL_QUALITY', 'balanced_q3')
UNET_QUALITY_FILES = {
    'fast_q2': 'ltx-2.3-22b-distilled-1.1-Q2_K.gguf',
    'balanced_q3': 'ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf',
    'quality_q4': 'ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf',
}
CLIP_QUALITY_FILES = {
    'fast_q2': 'gemma-3-12b-it-Q2_K.gguf',
    'balanced_q3': 'gemma-3-12b-it-Q2_K.gguf',
    'quality_q4': 'gemma-3-12b-it-Q3_K_M.gguf',
}
MODEL_FILES = {
    'unet': f"LTX2/{UNET_QUALITY_FILES[MODEL_QUALITY]}",
    'clip': CLIP_QUALITY_FILES[MODEL_QUALITY],
    'projection': 'ltx-2.3_text_projection_bf16.safetensors',
    'video_vae': 'LTX-2.3/LTX23_video_vae_bf16.safetensors',
    'audio_vae': 'LTX-2.3/LTX23_audio_vae_bf16.safetensors',
    'tiny_vae': 'LTX-2.3/taeltx2_3.safetensors',
    'upscaler': 'ltx-2.3-spatial-upscaler-x2-1.0.safetensors',
    'ic_union_lora': 'LTX-2.3/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors',
    'sdpose': 'SDPose/sdpose_wholebody_fp16.safetensors',
}
PRESETS = {
    'L4 fast preview - 512x896 49f': {'width': 512, 'height': 896, 'frames': 49, 'decode_tile': 512, 'temporal_size': 1024, 'chunk': 2, 'use_nag': False},
    'L4 balanced portrait - 640x1024 73f': {'width': 640, 'height': 1024, 'frames': 73, 'decode_tile': 512, 'temporal_size': 2048, 'chunk': 2, 'use_nag': False},
    'L4 safe - 768x384 73f': {'width': 768, 'height': 384, 'frames': 73, 'decode_tile': 512, 'temporal_size': 2048, 'chunk': 2, 'use_nag': False},
    'L4 quality - 896x512 97f': {'width': 896, 'height': 512, 'frames': 97, 'decode_tile': 512, 'temporal_size': 2048, 'chunk': 2, 'use_nag': False},
    'L4 quality portrait - 768x1280 73f': {'width': 768, 'height': 1280, 'frames': 73, 'decode_tile': 512, 'temporal_size': 2048, 'chunk': 2, 'use_nag': False},
    'A100 quality - 1280x640 193f': {'width': 1280, 'height': 640, 'frames': 193, 'decode_tile': 640, 'temporal_size': 4096, 'chunk': 4, 'use_nag': True},
    'A100 portrait - 768x1280 121f': {'width': 768, 'height': 1280, 'frames': 121, 'decode_tile': 640, 'temporal_size': 4096, 'chunk': 4, 'use_nag': True},
}
DEFAULT_PRESET = os.getenv('DEFAULT_PRESET', 'L4 balanced portrait - 640x1024 73f')
DEFAULT_NEGATIVE = 'blurry, distorted body, bad anatomy, broken limbs, extra limbs, fused fingers, mangled hands, bad teeth, distorted mouth, warped face, melted skin, flicker, jitter, low quality'
PASS1_SIGMAS = '1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0'
PASS2_SIGMAS = '0.85, 0.7250, 0.4219, 0.0'
LTX_IMAGE_COMPRESSION = 12
DEFAULT_FPS = int(os.getenv('DEFAULT_FPS', '12'))
LONG_VIDEO_MODE = os.getenv('LONG_VIDEO_MODE', 'true').lower() == 'true'
LONG_VIDEO_MAX_FRAMES = int(os.getenv('LONG_VIDEO_MAX_FRAMES', '721'))
DEFAULT_CHUNK = int(os.getenv('DEFAULT_CHUNK', '4'))
LONG_VIDEO_CHUNK = int(os.getenv('LONG_VIDEO_CHUNK', '8'))
DEFAULT_DECODE_TILE = int(os.getenv('DEFAULT_DECODE_TILE', '384'))
LONG_VIDEO_DECODE_TILE = int(os.getenv('LONG_VIDEO_DECODE_TILE', '256'))
DEFAULT_TEMPORAL_SIZE = int(os.getenv('DEFAULT_TEMPORAL_SIZE', '1024'))
LONG_VIDEO_TEMPORAL_SIZE = int(os.getenv('LONG_VIDEO_TEMPORAL_SIZE', '512'))

MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'media.aiclip.ai')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', '')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', '')
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'video')
MINIO_SECURE = os.getenv('MINIO_SECURE', 'false').lower() == 'true'
MINIO_PUBLIC_BASE_URL = os.getenv('MINIO_PUBLIC_BASE_URL', f"{'https' if MINIO_SECURE else 'http'}://{MINIO_ENDPOINT}")

BOOTSTRAPPED = False
NODE_CLASS_MAPPINGS = {}
MODEL_CACHE = {}

for path in (INPUT_DIR, OUTPUT_DIR, WORKSPACE_DIR / 'tmp'):
    path.mkdir(parents=True, exist_ok=True)

os.environ['AUX_ANNOTATOR_CKPTS_PATH'] = str(AUX_ANNOTATOR_DIR)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')


def download_file(url: str, output_path: Path, timeout: int = 600) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with output_path.open('wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return output_path


def validate_input(job_input: Dict[str, Any]) -> None:
    for field in ['source_video_url', 'control_video_url', 'prompt']:
        if not job_input.get(field):
            raise ValueError(f"Missing required field: {field}")
    for field in ('source_video_url', 'control_video_url'):
        if not str(job_input[field]).startswith(('http://', 'https://')):
            raise ValueError(f"{field} must be HTTP(S) URL")
    preset = job_input.get('preset')
    if preset and preset not in PRESETS:
        raise ValueError(f"Invalid preset: {preset}")


def get_minio_client() -> Minio:
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise ValueError('Missing MinIO credentials')
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)


def upload_to_minio(local_path: Path, object_name: str) -> str:
    client = get_minio_client()
    client.fput_object(MINIO_BUCKET, object_name, str(local_path))
    return f"{MINIO_PUBLIC_BASE_URL.rstrip('/')}/{MINIO_BUCKET}/{quote(object_name)}"


def random_seed() -> int:
    return random.randint(0, 2**32 - 1)


def get_value(obj: Any, index: int = 0) -> Any:
    try:
        return obj[index]
    except Exception:
        return obj['result'][index]



def require_node(*names: str) -> Tuple[str, Any]:
    for name in names:
        if name in NODE_CLASS_MAPPINGS:
            return name, NODE_CLASS_MAPPINGS[name]
    available = ', '.join(sorted(k for k in NODE_CLASS_MAPPINGS if any(token in k.lower() for token in ('vhs', 'video', 'load'))))
    raise KeyError(f"Missing required ComfyUI node. Tried {names}. Related available nodes: {available}")

def call_node(node: Any, **kwargs: Any) -> Any:
    fn_name = getattr(node, 'FUNCTION', None)
    if not fn_name:
        raise RuntimeError(f'{type(node).__name__} has no FUNCTION attribute')
    return getattr(node, fn_name)(**kwargs)


def clear_memory(unload_cache: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if unload_cache:
        MODEL_CACHE.clear()
        gc.collect()


def prepare_comfy_imports() -> None:
    # ComfyUI custom nodes import `utils.install_util`. Some base images or
    # dependencies also provide a top-level `utils.py`, so Python can resolve
    # `utils` to that module before it sees `/ComfyUI/utils`, producing:
    #   No module named 'utils.install_util'; 'utils' is not a package
    # Install a small compatibility package early and evict any stale module.
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))

    utils_path = COMFY_ROOT / 'utils'
    utils_path.mkdir(parents=True, exist_ok=True)
    (utils_path / '__init__.py').write_text('# generated RunPod compatibility package\n', encoding='utf-8')
    (utils_path / 'install_util.py').write_text(
        'from pathlib import Path\n\n'
        'def get_missing_requirements_message(packages):\n    return ""\n\n'
        'def get_required_packages_versions(requirements=None):\n    return {}\n\n'
        f'def requirements_path():\n    return str(Path(r"{COMFY_ROOT / "requirements.txt"}"))\n',
        encoding='utf-8',
    )

    import importlib
    import importlib.util
    import types

    # Force `utils` to be a package backed by /ComfyUI/utils even if another
    # dependency imported a plain top-level utils module first.
    pkg = types.ModuleType('utils')
    pkg.__file__ = str(utils_path / '__init__.py')
    pkg.__path__ = [str(utils_path)]
    pkg.__package__ = 'utils'
    sys.modules['utils'] = pkg

    install_spec = importlib.util.spec_from_file_location('utils.install_util', utils_path / 'install_util.py')
    if install_spec is None or install_spec.loader is None:
        raise RuntimeError(f'Could not create import spec for {utils_path / "install_util.py"}')
    install_module = importlib.util.module_from_spec(install_spec)
    sys.modules['utils.install_util'] = install_module
    install_spec.loader.exec_module(install_module)

    importlib.invalidate_caches()
    try:
        import utils.install_util  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f'Failed to prepare ComfyUI utils.install_util shim at {utils_path}: {exc}') from exc



def run_coro_sync(coro):
    """Run an async ComfyUI initializer from RunPod's possibly-active loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # RunPod invokes sync handlers inside an active event loop in some workers.
    # A helper thread gives the coroutine a clean loop and avoids nested-loop errors.
    import threading
    result = {}

    def runner():
        try:
            result['value'] = asyncio.run(coro)
        except BaseException as exc:
            result['error'] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if 'error' in result:
        raise result['error']
    return result.get('value')

def import_custom_nodes() -> None:
    global NODE_CLASS_MAPPINGS
    import execution  # noqa
    import folder_paths  # noqa
    import server  # noqa
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes, NODE_CLASS_MAPPINGS as NCM

    custom_nodes_dir = str(COMFY_ROOT / 'custom_nodes')
    existing_custom_paths = [str(Path(p).resolve()) for p in folder_paths.get_folder_paths('custom_nodes')]
    if str(Path(custom_nodes_dir).resolve()) not in existing_custom_paths:
        folder_paths.add_model_folder_path('custom_nodes', custom_nodes_dir)
    folder_paths.set_input_directory(str(INPUT_DIR))
    folder_paths.set_output_directory(str(OUTPUT_DIR))

    class DummyPromptServer:
        instance = None
        def __init__(self):
            DummyPromptServer.instance = self
            self.routes = None
            self.client_id = 'runpod-headless'
        def send_sync(self, *args, **kwargs):
            return None

    server.PromptServer = DummyPromptServer
    DummyPromptServer()

    # Current ComfyUI exposes async node initializers. Calling them without
    # awaiting silently leaves custom nodes unloaded, so only core nodes appear.
    run_coro_sync(init_builtin_extra_nodes())
    run_coro_sync(init_external_custom_nodes())
    NODE_CLASS_MAPPINGS = NCM
    logger.info(
        'Loaded ComfyUI nodes=%s; custom/video nodes=%s',
        len(NODE_CLASS_MAPPINGS),
        sorted(k for k in NODE_CLASS_MAPPINGS if any(token in k.lower() for token in ('vhs', 'video', 'ltx', 'gguf', 'kj')))[:80],
    )


def bootstrap_environment() -> None:
    global BOOTSTRAPPED
    if BOOTSTRAPPED:
        return
    prepare_comfy_imports()
    import_custom_nodes()
    BOOTSTRAPPED = True
    logger.info('Bootstrap done, nodes=%s', len(NODE_CLASS_MAPPINGS))


def sampler_select(name: str, fallbacks=('euler_ancestral_cfg_pp', 'euler_ancestral', 'euler')):
    node = NODE_CLASS_MAPPINGS['KSamplerSelect']()
    errors = []
    for sampler_name in (name,) + tuple(item for item in fallbacks if item != name):
        try:
            return get_value(call_node(node, sampler_name=sampler_name), 0), sampler_name
        except Exception as exc:
            errors.append(f'{sampler_name}: {exc}')
    raise RuntimeError('No sampler worked: ' + '; '.join(errors))


def _round_to_multiple(value, multiple=64, minimum=64):
    return max(minimum, int(math.ceil(float(value) / multiple) * multiple))


def probe_video_metadata(video_path: Path) -> Dict[str, Any]:
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,duration:format=duration',
        '-of', 'json', str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout or '{}')
    stream = (data.get('streams') or [{}])[0]
    fmt = data.get('format') or {}
    fps_raw = stream.get('avg_frame_rate', '0/1')
    fps = 0.0
    if isinstance(fps_raw, str) and '/' in fps_raw:
        a, b = fps_raw.split('/', 1)
        try:
            fps = float(a) / float(b)
        except Exception:
            fps = 0.0
    duration = float(stream.get('duration') or fmt.get('duration') or 0.0)
    return {
        'width': int(stream.get('width') or 0),
        'height': int(stream.get('height') or 0),
        'fps': fps,
        'duration': duration,
    }


def suggest_target_size(ctrl_w: int, ctrl_h: int, long_video_mode: bool = True) -> tuple[int, int]:
    portrait = ctrl_h >= ctrl_w
    if portrait:
        target_h = 896 if long_video_mode else 1024
        scale = target_h / max(ctrl_h, 1)
        target_w = _round_to_multiple(ctrl_w * scale, 64, 320)
        target_w = min(target_w, 640 if long_video_mode else 768)
        target_h = _round_to_multiple(target_h, 64, 640)
    else:
        target_w = 768 if long_video_mode else 896
        scale = target_w / max(ctrl_w, 1)
        target_h = _round_to_multiple(ctrl_h * scale, 64, 320)
        target_h = min(target_h, 512 if long_video_mode else 640)
        target_w = _round_to_multiple(target_w, 64, 640)
    return int(target_w), int(target_h)


def resize_crop_images_torch(images, target_w, target_h, mode='bicubic'):
    import torch.nn.functional as F
    b, h, w, c = images.shape
    img = images.permute(0, 3, 1, 2)
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = F.interpolate(img, size=(new_h, new_w), mode=mode, align_corners=False if mode in ('bilinear', 'bicubic') else None)
    top = max(0, (new_h - target_h) // 2)
    left = max(0, (new_w - target_w) // 2)
    img = img[:, :, top:top+target_h, left:left+target_w]
    return img.permute(0, 2, 3, 1).contiguous()


def resize_shorter_torch(images, shorter=1024, mode='bicubic'):
    import torch.nn.functional as F
    b, h, w, c = images.shape
    scale = shorter / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = images.permute(0, 3, 1, 2)
    img = F.interpolate(img, size=(nh, nw), mode=mode, align_corners=False if mode in ('bilinear', 'bicubic') else None)
    return img.permute(0, 2, 3, 1).contiguous()


def blend_multiply_pose_depth(pose_img, depth_img, blend_factor=0.5):
    return pose_img * (1.0 - float(blend_factor)) + depth_img * float(blend_factor)


def save_video_with_optional_audio(frames, audio, fps: int, output_path: Path):
    import av
    import numpy as np
    frames_np = (frames.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output_path), mode='w')
    stream = container.add_stream('libx264', rate=int(fps))
    stream.width = int(frames_np.shape[2])
    stream.height = int(frames_np.shape[1])
    stream.pix_fmt = 'yuv420p'
    for frame in frames_np:
        video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def load_body_transfer_components(ic_strength=0.71, chunk=2):
    key = (round(float(ic_strength), 3), int(chunk))
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    clear_memory()
    model = get_value(call_node(NODE_CLASS_MAPPINGS['UnetLoaderGGUF'](), unet_name=MODEL_FILES['unet']), 0)
    ic_out = call_node(NODE_CLASS_MAPPINGS['LTXICLoRALoaderModelOnly'](), model=model, lora_name=MODEL_FILES['ic_union_lora'], strength_model=float(ic_strength))
    model = get_value(ic_out, 0)
    latent_downscale_factor = get_value(ic_out, 1)
    clip = get_value(call_node(NODE_CLASS_MAPPINGS['DualCLIPLoaderGGUF'](), clip_name1=MODEL_FILES['clip'], clip_name2=MODEL_FILES['projection'], type='ltxv'), 0)
    vae_video = get_value(call_node(NODE_CLASS_MAPPINGS['VAELoader'](), vae_name=MODEL_FILES['video_vae']), 0)
    vae_audio = get_value(call_node(NODE_CLASS_MAPPINGS.get('VAELoaderKJ', NODE_CLASS_MAPPINGS['VAELoader'])(), vae_name=MODEL_FILES['audio_vae'], device='main_device', weight_dtype='bf16') if 'VAELoaderKJ' in NODE_CLASS_MAPPINGS else call_node(NODE_CLASS_MAPPINGS['VAELoader'](), vae_name=MODEL_FILES['audio_vae']), 0)
    up_model = get_value(call_node(NODE_CLASS_MAPPINGS['LatentUpscaleModelLoader'](), model_name=MODEL_FILES['upscaler']), 0)
    MODEL_CACHE[key] = {'model': model, 'clip': clip, 'vae_video': vae_video, 'vae_audio': vae_audio, 'up_model': up_model, 'latent_downscale_factor': latent_downscale_factor}
    return MODEL_CACHE[key]


def extract_control_guide(control_frames, pose_method='DWPose', blend_pose_depth=False, blend_factor=0.5, pass_width=480, pass_height=272):
    preprocess = NODE_CLASS_MAPPINGS['LTXVPreprocess']()
    control_small = resize_crop_images_torch(control_frames, pass_width, pass_height, mode='bicubic')
    control_pp = get_value(call_node(preprocess, image=control_small, img_compression=LTX_IMAGE_COMPRESSION), 0)
    pose_method_norm = str(pose_method or 'DWPose').lower()
    if pose_method_norm.startswith('sd'):
        sd_img = resize_shorter_torch(control_pp, 1024, mode='bicubic')
        sdpose_out = call_node(NODE_CLASS_MAPPINGS['CheckpointLoaderSimple'](), ckpt_name=MODEL_FILES['sdpose'])
        ckpt = get_value(sdpose_out, 0)
        ckpt_vae = get_value(sdpose_out, 2)
        kps = get_value(call_node(NODE_CLASS_MAPPINGS['SDPoseKeypointExtractor'](), model=ckpt, vae=ckpt_vae, image=sd_img, batch_size=16), 0)
        pose_img = get_value(call_node(NODE_CLASS_MAPPINGS['SDPoseDrawKeypoints'](), image=control_pp, keypoints=kps), 0)
    else:
        pose_img = get_value(call_node(NODE_CLASS_MAPPINGS['DWPreprocessor'](), image=control_pp, detect_hand=True, detect_body=True, detect_face=True, resolution=768, bbox_detector='yolox_l.onnx', pose_estimator='dw-ll_ucoco_384.onnx', scale_stick_for_xinsr_cn=False), 0)
    if blend_pose_depth and 'DepthAnythingPreprocessor' in NODE_CLASS_MAPPINGS:
        depth_img = get_value(call_node(NODE_CLASS_MAPPINGS['DepthAnythingPreprocessor'](), image=control_pp, ckpt_name='depth_anything_vitl14.pth'), 0)
        pose_img = blend_multiply_pose_depth(pose_img, depth_img, blend_factor=blend_factor)
    return pose_img


def run_body_transfer(source_video_path: Path, control_video_path: Path, prompt: str, negative_prompt: str = DEFAULT_NEGATIVE, preset: str | None = None, pose_method: str = 'DWPose', blend_pose_depth: bool = False, blend_factor: float = 0.5, guide_strength: float = 0.8, ic_strength: float = 0.71, cfg: float = 1.0, seed: int = -1, fps: int = DEFAULT_FPS, use_control_audio: bool = True, use_nag=None, nag_scale: float = 11.0, nag_alpha: float = 0.25, nag_tau: float = 2.5, sampler_pass1: str = 'euler_ancestral_cfg_pp', sampler_pass2: str = 'euler_cfg_pp', unload_after: bool = False, output_width: int = 0, output_height: int = 0, match_control_orientation: bool = True, max_frames: int = LONG_VIDEO_MAX_FRAMES, chunk: int | None = None, decode_tile: int | None = None, temporal_size: int | None = None, long_video_mode: bool = LONG_VIDEO_MODE) -> Path:
    bootstrap_environment()
    if seed is None or int(seed) < 0:
        seed = random_seed()
    src_name = Path(source_video_path).name
    ctrl_name = Path(control_video_path).name
    src_in = INPUT_DIR / src_name
    ctrl_in = INPUT_DIR / ctrl_name
    if Path(source_video_path).resolve() != src_in.resolve():
        shutil.copy(source_video_path, src_in)
    if Path(control_video_path).resolve() != ctrl_in.resolve():
        shutil.copy(control_video_path, ctrl_in)

    ctrl_meta = probe_video_metadata(ctrl_in)
    ctrl_w_meta = int(ctrl_meta.get('width') or 0)
    ctrl_h_meta = int(ctrl_meta.get('height') or 0)
    src_meta = probe_video_metadata(src_in)
    requested_fps = int(fps or 0) if fps is not None else 0
    native_fps = int(round(ctrl_meta.get('fps') or 0))
    target_fps = requested_fps if requested_fps > 0 else (native_fps if native_fps > 0 else DEFAULT_FPS)
    if long_video_mode:
        target_fps = min(target_fps, DEFAULT_FPS)

    if preset:
        preset_data = dict(PRESETS[preset])
        auto_width = int(preset_data['width'])
        auto_height = int(preset_data['height'])
        decode_tile_final = int(decode_tile or preset_data.get('decode_tile', DEFAULT_DECODE_TILE))
        temporal_size_final = int(temporal_size or preset_data.get('temporal_size', DEFAULT_TEMPORAL_SIZE))
        chunk_final = int(chunk or preset_data.get('chunk', DEFAULT_CHUNK))
        default_use_nag = bool(preset_data.get('use_nag', False))
    else:
        auto_width, auto_height = suggest_target_size(ctrl_w_meta, ctrl_h_meta, long_video_mode=long_video_mode)
        decode_tile_final = int(decode_tile or (LONG_VIDEO_DECODE_TILE if long_video_mode else DEFAULT_DECODE_TILE))
        temporal_size_final = int(temporal_size or (LONG_VIDEO_TEMPORAL_SIZE if long_video_mode else DEFAULT_TEMPORAL_SIZE))
        chunk_final = int(chunk or (LONG_VIDEO_CHUNK if long_video_mode else DEFAULT_CHUNK))
        default_use_nag = False

    target_width = int(output_width) if output_width and int(output_width) > 0 else auto_width
    target_height = int(output_height) if output_height and int(output_height) > 0 else auto_height
    if use_nag is None:
        use_nag = default_use_nag

    with torch.inference_mode():
        vloader_name, vloader_cls = require_node('VHS_LoadVideoFFmpeg', 'VHS_LoadVideo', 'LoadVideo')
        logger.info('Using video loader node: %s', vloader_name)
        vloader = vloader_cls()
        src = call_node(vloader, video=src_name, force_rate=float(target_fps), custom_width=0, custom_height=0, frame_load_cap=int(max_frames), start_time=0, format='LTXV')
        ctrl = call_node(vloader, video=ctrl_name, force_rate=float(target_fps), custom_width=0, custom_height=0, frame_load_cap=int(max_frames), start_time=0, format='LTXV')
        src_frames_all = get_value(src, 0)
        ctrl_frames_all = get_value(ctrl, 0)
        ctrl_audio = get_value(ctrl, 2) if use_control_audio else None
        if int(ctrl_frames_all.shape[0]) < 9:
            raise RuntimeError('Control video must contain at least 9 frames after FPS/frame cap conversion.')
        src_h, src_w = int(src_frames_all.shape[1]), int(src_frames_all.shape[2])
        ctrl_h, ctrl_w = int(ctrl_frames_all.shape[1]), int(ctrl_frames_all.shape[2])
        if match_control_orientation and ((ctrl_w >= ctrl_h) != (target_width >= target_height)):
            target_width, target_height = target_height, target_width
            logger.info('Matched control orientation: target swapped to %sx%s', target_width, target_height)
        pass_w = _round_to_multiple(target_width / 2, 64, 64)
        pass_h = _round_to_multiple(target_height / 2, 64, 64)
        width, height = pass_w * 2, pass_h * 2
        frames = ((min(int(ctrl_frames_all.shape[0]), int(max_frames)) - 1) // 8) * 8 + 1
        source_ref = src_frames_all[:1]
        ctrl_frames = ctrl_frames_all[:frames]
        logger.info('Seed=%s source=%sx%s control=%sx%s native_ctrl=%sx%s duration=%.2fs fps=%s frames=%s output=%sx%s chunk=%s decode_tile=%s temporal_size=%s long_video=%s', seed, src_w, src_h, ctrl_w, ctrl_h, ctrl_w_meta, ctrl_h_meta, float(ctrl_meta.get('duration') or 0.0), target_fps, frames, width, height, chunk_final, decode_tile_final, temporal_size_final, long_video_mode)
        comp = load_body_transfer_components(ic_strength=ic_strength, chunk=chunk_final)
        preprocess = NODE_CLASS_MAPPINGS['LTXVPreprocess']()
        src_pass = resize_crop_images_torch(source_ref, pass_w, pass_h, mode='bicubic')
        src_pp = get_value(call_node(preprocess, image=src_pass, img_compression=LTX_IMAGE_COMPRESSION), 0)
        guide_img = extract_control_guide(ctrl_frames, pose_method=pose_method, blend_pose_depth=blend_pose_depth, blend_factor=blend_factor, pass_width=pass_w, pass_height=pass_h)
        te = NODE_CLASS_MAPPINGS['CLIPTextEncode']()
        pos = get_value(te.encode(text=prompt, clip=comp['clip']), 0)
        neg = get_value(te.encode(text=negative_prompt, clip=comp['clip']), 0)
        cond = call_node(NODE_CLASS_MAPPINGS['LTXVConditioning'](), positive=pos, negative=neg, frame_rate=float(target_fps))
        pos_cond, neg_cond = get_value(cond, 0), get_value(cond, 1)
        model = comp['model']
        if use_nag and 'LTX2_NAG' in NODE_CLASS_MAPPINGS:
            model = get_value(call_node(NODE_CLASS_MAPPINGS['LTX2_NAG'](), model=model, nag_scale=float(nag_scale), nag_alpha=float(nag_alpha), nag_tau=float(nag_tau), nag_cond_video=neg_cond, nag_cond_audio=neg_cond, inplace=True), 0)
        empty_video = get_value(call_node(NODE_CLASS_MAPPINGS['EmptyLTXVLatentVideo'](), width=pass_w, height=pass_h, length=frames, batch_size=1), 0)
        i2v_cond = get_value(call_node(NODE_CLASS_MAPPINGS['LTXVImgToVideoConditionOnly'](), vae=comp['vae_video'], image=src_pp, latent=empty_video, strength=1.0, bypass=False), 0)
        guided = call_node(NODE_CLASS_MAPPINGS['LTXAddVideoICLoRAGuide'](), positive=pos_cond, negative=neg_cond, vae=comp['vae_video'], latent=i2v_cond, image=guide_img, frame_idx=0, strength=float(guide_strength), latent_downscale_factor=comp['latent_downscale_factor'], crop='disabled', use_tiled_encode=False, tile_size=256, tile_overlap=64)
        pos_guided, neg_guided, video_latent = get_value(guided, 0), get_value(guided, 1), get_value(guided, 2)
        if use_control_audio and ctrl_audio is not None:
            audio_latent = get_value(call_node(NODE_CLASS_MAPPINGS['LTXVAudioVAEEncode'](), audio=ctrl_audio, audio_vae=comp['vae_audio']), 0)
        else:
            audio_latent = get_value(call_node(NODE_CLASS_MAPPINGS['LTXVEmptyLatentAudio'](), audio_vae=comp['vae_audio'], frames_number=frames, frame_rate=int(target_fps), batch_size=1), 0)
        concat = NODE_CLASS_MAPPINGS['LTXVConcatAVLatent']()
        av_latent = get_value(call_node(concat, video_latent=video_latent, audio_latent=audio_latent), 0)
        sampler1, _ = sampler_select(sampler_pass1, ('euler_ancestral_cfg_pp', 'euler_ancestral', 'euler'))
        sigmas1 = get_value(call_node(NODE_CLASS_MAPPINGS['ManualSigmas'](), sigmas=PASS1_SIGMAS), 0)
        guider1 = get_value(NODE_CLASS_MAPPINGS['CFGGuider']().get_guider(model=model, positive=pos_guided, negative=neg_guided, cfg=float(cfg)), 0)
        noise1 = get_value(NODE_CLASS_MAPPINGS['RandomNoise']().get_noise(noise_seed=int(seed)), 0)
        sample1 = get_value(NODE_CLASS_MAPPINGS['SamplerCustomAdvanced']().sample(noise=noise1, guider=guider1, sampler=sampler1, sigmas=sigmas1, latent_image=av_latent), 0)
        sep1 = call_node(NODE_CLASS_MAPPINGS['LTXVSeparateAVLatent'](), av_latent=sample1)
        video1, audio1 = get_value(sep1, 0), get_value(sep1, 1)
        crop = call_node(NODE_CLASS_MAPPINGS['LTXVCropGuides'](), positive=pos_guided, negative=neg_guided, latent=video1)
        crop_pos, crop_neg, crop_video = get_value(crop, 0), get_value(crop, 1), get_value(crop, 2)
        clear_memory()
        up_video = get_value(call_node(NODE_CLASS_MAPPINGS['LTXVLatentUpsampler'](), samples=crop_video, upscale_model=comp['up_model'], vae=comp['vae_video']), 0)
        src_full = resize_crop_images_torch(source_ref, width, height, mode='bicubic')
        src_full_pp = get_value(call_node(preprocess, image=src_full, img_compression=LTX_IMAGE_COMPRESSION), 0)
        up_video = get_value(call_node(NODE_CLASS_MAPPINGS['LTXVImgToVideoInplace'](), vae=comp['vae_video'], image=src_full_pp, latent=up_video, strength=0.7, bypass=False), 0)
        av_latent2 = get_value(call_node(concat, video_latent=up_video, audio_latent=audio1), 0)
        sampler2, _ = sampler_select(sampler_pass2, ('euler_cfg_pp', 'euler', 'euler_ancestral'))
        sigmas2 = get_value(call_node(NODE_CLASS_MAPPINGS['ManualSigmas'](), sigmas=PASS2_SIGMAS), 0)
        guider2 = get_value(NODE_CLASS_MAPPINGS['CFGGuider']().get_guider(model=model, positive=crop_pos, negative=crop_neg, cfg=float(cfg)), 0)
        noise2 = get_value(NODE_CLASS_MAPPINGS['RandomNoise']().get_noise(noise_seed=int(seed) + 1), 0)
        sample2 = get_value(NODE_CLASS_MAPPINGS['SamplerCustomAdvanced']().sample(noise=noise2, guider=guider2, sampler=sampler2, sigmas=sigmas2, latent_image=av_latent2), 0)
        sep2 = call_node(NODE_CLASS_MAPPINGS['LTXVSeparateAVLatent'](), av_latent=sample2)
        video2, _audio2 = get_value(sep2, 0), get_value(sep2, 1)
        crop_final = get_value(call_node(NODE_CLASS_MAPPINGS['LTXVCropGuides'](), positive=pos_cond, negative=neg_cond, latent=video2), 2)
        decoded = get_value(call_node(NODE_CLASS_MAPPINGS['VAEDecodeTiled'](), samples=crop_final, vae=comp['vae_video'], tile_size=int(decode_tile_final), overlap=64, temporal_size=int(temporal_size_final), temporal_overlap=8), 0)
        out = OUTPUT_DIR / f'body_transfer_{seed}.mp4'
        save_video_with_optional_audio(decoded, ctrl_audio if use_control_audio and ctrl_audio is not None else None, int(target_fps), out)
    clear_memory(unload_cache=bool(unload_after))
    return out


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    start_time = time.time()
    job_input = job.get('input', {})
    job_id = job.get('id', f'job-{int(start_time)}')
    logger.info('Starting job %s input=%s', job_id, json.dumps(job_input, ensure_ascii=False))
    source_path = None
    control_path = None
    try:
        validate_input(job_input)
        source_suffix = Path(job_input['source_video_url']).suffix or '.mp4'
        control_suffix = Path(job_input['control_video_url']).suffix or '.mp4'
        source_path = download_file(job_input['source_video_url'], INPUT_DIR / f"{job_id}_source{source_suffix}")
        control_path = download_file(job_input['control_video_url'], INPUT_DIR / f"{job_id}_control{control_suffix}")
        selected_preset = job_input.get('preset')
        long_video_mode = bool(job_input.get('long_video_mode', LONG_VIDEO_MODE))
        ctrl_meta = probe_video_metadata(control_path)
        if selected_preset:
            preset_config = PRESETS[selected_preset]
            resolved_width = int(job_input.get('output_width', 0)) or int(preset_config['width'])
            resolved_height = int(job_input.get('output_height', 0)) or int(preset_config['height'])
        else:
            resolved_width = int(job_input.get('output_width', 0)) or suggest_target_size(int(ctrl_meta.get('width') or 0), int(ctrl_meta.get('height') or 0), long_video_mode=long_video_mode)[0]
            resolved_height = int(job_input.get('output_height', 0)) or suggest_target_size(int(ctrl_meta.get('width') or 0), int(ctrl_meta.get('height') or 0), long_video_mode=long_video_mode)[1]
        output_path = run_body_transfer(
            source_video_path=source_path,
            control_video_path=control_path,
            prompt=job_input['prompt'],
            negative_prompt=job_input.get('negative_prompt', DEFAULT_NEGATIVE),
            preset=selected_preset,
            pose_method=job_input.get('pose_method', 'DWPose'),
            blend_pose_depth=job_input.get('blend_pose_depth', False),
            blend_factor=float(job_input.get('blend_factor', 0.5)),
            guide_strength=float(job_input.get('guide_strength', 0.8)),
            ic_strength=float(job_input.get('ic_strength', 0.71)),
            cfg=float(job_input.get('cfg', 1.0)),
            seed=int(job_input.get('seed', -1)),
            fps=int(job_input.get('fps', DEFAULT_FPS)),
            use_control_audio=job_input.get('use_control_audio', True),
            use_nag=job_input.get('use_nag', None),
            nag_scale=float(job_input.get('nag_scale', 11.0)),
            nag_alpha=float(job_input.get('nag_alpha', 0.25)),
            nag_tau=float(job_input.get('nag_tau', 2.5)),
            sampler_pass1=job_input.get('sampler_pass1', 'euler_ancestral_cfg_pp'),
            sampler_pass2=job_input.get('sampler_pass2', 'euler_cfg_pp'),
            unload_after=job_input.get('unload_after', False),
            output_width=int(job_input.get('output_width', 0)),
            output_height=int(job_input.get('output_height', 0)),
            match_control_orientation=job_input.get('match_control_orientation', True),
            max_frames=int(job_input.get('max_frames', LONG_VIDEO_MAX_FRAMES)),
            chunk=(int(job_input['chunk']) if job_input.get('chunk') is not None else None),
            decode_tile=(int(job_input['decode_tile']) if job_input.get('decode_tile') is not None else None),
            temporal_size=(int(job_input['temporal_size']) if job_input.get('temporal_size') is not None else None),
            long_video_mode=long_video_mode,
        )
        object_name = job_input.get('output_key', f'body-transfer/{job_id}/{output_path.name}')
        video_url = upload_to_minio(output_path, object_name)
        elapsed = round(time.time() - start_time, 2)
        return {
            'status': 'success',
            'output': {
                'video_url': video_url,
                'filename': output_path.name,
                'fps': int(job_input.get('fps', DEFAULT_FPS)),
                'resolution': job_input.get('resolution', f'{resolved_width}x{resolved_height}'),
                'file_size': output_path.stat().st_size,
            },
            'metadata': {
                'job_id': job_id,
                'processing_time': elapsed,
                'preset': selected_preset,
                'long_video_mode': long_video_mode,
                'control_duration': float(ctrl_meta.get('duration') or 0.0),
            }
        }
    except Exception as exc:
        logger.exception('Job failed: %s', exc)
        return {'status': 'error', 'error': str(exc), 'job_id': job_id}
    finally:
        for path in (source_path, control_path):
            try:
                if path and Path(path).exists():
                    Path(path).unlink()
            except Exception:
                pass


if __name__ == '__main__':
    logger.info('Starting RunPod serverless handler')
    runpod.serverless.start({'handler': handler})
