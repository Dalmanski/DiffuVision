import urllib.request
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from modules.sd_ideal import limit_longest_side
from utils.model_utils import clear_gpu
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except Exception:
    RRDBNet = RealESRGANer = None
BASE_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
ESRGAN_PATH = BASE_DIR / "weights" / "RealESRGAN_x4plus.pth"
_esrgan = None

def cleanup():
    global _esrgan
    _esrgan = None
    clear_gpu(synchronize=True)

def load_esrgan(logger=None):
    global _esrgan
    if _esrgan is not None:
        return _esrgan
    if not RRDBNet or not RealESRGANer:
        raise RuntimeError("Real-ESRGAN dependencies are not installed.")
    ESRGAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ESRGAN_PATH.exists():
        if logger:
            logger("Downloading Real-ESRGAN model...")
        urllib.request.urlretrieve(ESRGAN_URL, str(ESRGAN_PATH))
    if logger:
        logger("Loading Real-ESRGAN...")
    _esrgan = RealESRGANer(scale=4, model_path=str(ESRGAN_PATH), model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), tile=256, tile_pad=10, pre_pad=0, half=DEVICE == "cuda", gpu_id=0 if DEVICE == "cuda" else None)
    return _esrgan

def enhance(image, output=False, logger=None, output_target=1080):
    esrgan = load_esrgan(logger)
    if logger:
        logger("Enhancing output with Real-ESRGAN..." if output else "Enhancing original with Real-ESRGAN...")
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    out, _ = esrgan.enhance(bgr, outscale=4)
    cleanup()
    result = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).convert("RGB")
    if output:
        result = limit_longest_side(result, output_target)
        if logger:
            logger(f"Output enhancement complete • {result.width}x{result.height}")
    return result