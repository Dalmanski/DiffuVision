import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.image_loader import pixel_box
from utils.model_utils import clear_gpu

SAM2_CKPT_NAME = "sam2.1_hiera_small.pt"
SAM2_CKPT_PATH = Path("checkpoints") / SAM2_CKPT_NAME
SAM2_CONFIG_NAME = "sam2.1_hiera_s.yaml"
SAM2_CONFIG_PATH = Path("sam2") / "configs" / "sam2.1" / SAM2_CONFIG_NAME
SMALLEST_SCORE_RATIO = 0.80

def _has_sam2_files(directory):
    return (directory / SAM2_CKPT_PATH).is_file() and (directory / SAM2_CONFIG_PATH).is_file()

class SAM2Segmenter:
    def __init__(self, repo_dir=None):
        self.repo_dir = Path(repo_dir).resolve() if repo_dir else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.predictor = None
        self._reset_state(None)

    def _reset_state(self, image):
        self.image = image
        self.image_np = None if image is None else np.array(image)
        self.current_mask = None
        self.accumulated_mask = None if image is None else np.zeros((image.height, image.width), dtype=bool)
        self.segments = []
        self.prompt_points = []
        self.prompt_labels = []

    def find_repo_dir(self):
        if self.repo_dir is not None and _has_sam2_files(self.repo_dir):
            return self.repo_dir
        base_dir = Path(__file__).resolve().parents[1]
        env_value = os.environ.get("SAM2_REPO_DIR", "").strip()
        candidates = []
        if env_value:
            candidate = Path(env_value)
            candidates.append(candidate if candidate.is_absolute() else base_dir / candidate)
        candidates.extend([base_dir, Path(__file__).resolve().parent, base_dir / "model", base_dir / "models", base_dir.parent, base_dir.parent / "sam2", base_dir.parent / "model", base_dir.parent / "models"])
        candidates = [candidate.resolve(strict=False) for candidate in candidates]
        for candidate in candidates:
            if _has_sam2_files(candidate):
                self.repo_dir = candidate
                return candidate
        for root in candidates:
            if not root.exists():
                continue
            try:
                for checkpoint in root.rglob(SAM2_CKPT_NAME):
                    parent = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
                    if (parent / SAM2_CONFIG_PATH).is_file():
                        self.repo_dir = parent
                        return parent
            except OSError:
                continue
        raise FileNotFoundError(f"SAM 2 repository not found. Set SAM2_REPO_DIR to the folder containing {SAM2_CKPT_NAME} and {SAM2_CONFIG_NAME}.")

    def load(self):
        if self.model is not None:
            return
        if self.device != "cuda":
            raise RuntimeError("CUDA GPU is required for SAM 2.")
        repo_dir = self.find_repo_dir()
        print(f"Loading SAM 2 ({SAM2_CKPT_NAME}, {SAM2_CONFIG_NAME})...")
        self.model = build_sam2(str(repo_dir / SAM2_CONFIG_PATH), str(repo_dir / SAM2_CKPT_PATH), device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        print(f"SAM 2 ({SAM2_CKPT_NAME}, {SAM2_CONFIG_NAME}) loaded.")

    def load_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError("Image must be a file path or PIL Image.")
        self._reset_state(image)
        if self.predictor is None:
            self.load()
        self.predictor.set_image(self.image_np)

    def add_point(self, x, y, positive=True):
        if self.image is None:
            raise RuntimeError("No image has been loaded.")
        x = int(x)
        y = int(y)
        if x < 0 or y < 0 or x >= self.image.width or y >= self.image.height:
            return
        self.prompt_points.append((x, y))
        self.prompt_labels.append(1 if positive else 0)

    def clear_points(self):
        self.prompt_points.clear()
        self.prompt_labels.clear()
        self.current_mask = None

    def predict(self, prefer_smallest=True, mask_index=None):
        if self.predictor is None:
            raise RuntimeError("SAM 2 model is not loaded.")
        if self.image is None:
            raise RuntimeError("No image has been loaded.")
        if not self.prompt_points:
            raise RuntimeError("At least one prompt point is required.")
        points = np.asarray(self.prompt_points, dtype=np.float32)
        labels = np.asarray(self.prompt_labels, dtype=np.int32)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, scores, _ = self.predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
        masks = masks.astype(bool)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if mask_index is not None:
            best_index = int(mask_index)
        elif prefer_smallest:
            areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
            candidates = np.flatnonzero((scores >= scores.max() * SMALLEST_SCORE_RATIO) & (areas > 0))
            best_index = int(candidates[np.argmin(areas[candidates])]) if len(candidates) else int(np.argmax(scores))
        else:
            best_index = int(np.argmax(scores))
        self.current_mask = masks[best_index]
        return self.current_mask, float(scores[best_index])

    def append_segment(self):
        if self.current_mask is None:
            raise RuntimeError("No current segment is available.")
        if self.accumulated_mask is None:
            self.accumulated_mask = np.zeros_like(self.current_mask, dtype=bool)
        appended_mask = self.current_mask.copy()
        self.segments.append(appended_mask)
        self.accumulated_mask |= appended_mask
        self.clear_points()
        return appended_mask

    def remove_segment_at(self, x, y):
        for index in range(len(self.segments) - 1, -1, -1):
            if self.segments[index][int(y), int(x)]:
                self.segments.pop(index)
                self.accumulated_mask = np.zeros_like(self.accumulated_mask, dtype=bool)
                for segment in self.segments:
                    self.accumulated_mask |= segment
                return True
        return False

    def add_point_segment(self, x, y, positive=True, prefer_smallest=True, mask_index=None):
        self.add_point(x, y, positive)
        self.predict(prefer_smallest=prefer_smallest, mask_index=mask_index)
        self.append_segment()
        return self.get_mask()

    def build_mask(self, target_size=None, crop_box=None, original_size=None):
        accumulated = self.get_mask()
        if accumulated is None or not np.any(accumulated):
            raise RuntimeError("No accumulated selection is available.")
        raw_mask = Image.fromarray((accumulated.astype(np.uint8) * 255), mode="L")
        if original_size is not None and crop_box is not None:
            raw_mask = raw_mask.crop(pixel_box([float(value) for value in crop_box], original_size))
        if target_size is not None and raw_mask.size != tuple(target_size):
            raw_mask = raw_mask.resize(tuple(target_size), Image.Resampling.NEAREST)
        return raw_mask.copy()

    def select_point(self, x, y, positive=True, target_size=None, crop_box=None, original_size=None, prefer_smallest=True, mask_index=None):
        self.add_point_segment(x, y, positive, prefer_smallest=prefer_smallest, mask_index=mask_index)
        return self.build_mask(target_size=target_size, crop_box=crop_box, original_size=original_size)

    def clear_all_segments(self):
        self.clear_selection()
        return self.get_mask()

    def clear_selection(self):
        if self.image is None:
            self.accumulated_mask = None
        else:
            self.accumulated_mask = np.zeros((self.image.height, self.image.width), dtype=bool)
        self.segments = []
        self.clear_points()

    def get_mask(self):
        return None if self.accumulated_mask is None else self.accumulated_mask.copy()

    def get_image_size(self):
        if self.image is None:
            return None
        return self.image.size

    def close(self):
        try:
            if self.predictor is not None:
                self.predictor.reset_predictor()
        except Exception:
            pass
        self.predictor = None
        self.model = None
        self._reset_state(None)
        clear_gpu()
