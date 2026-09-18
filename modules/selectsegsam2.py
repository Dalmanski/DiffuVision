import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_CHECKPOINT_NAME = "sam2.1_hiera_small.pt"
SAM2_CHECKPOINT_PATH = Path("checkpoints") / SAM2_CHECKPOINT_NAME
SAM2_CONFIG_NAME = "sam2.1_hiera_s.yaml"

class SAM2Segmenter:
    def __init__(self, repo_dir=None):
        self.repo_dir = Path(repo_dir).resolve() if repo_dir else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.predictor = None
        self.image = None
        self.image_np = None
        self.current_mask = None
        self.accumulated_mask = None
        self.prompt_points = []
        self.prompt_labels = []

    def find_repo_dir(self):
        if self.repo_dir is not None and (self.repo_dir / SAM2_CHECKPOINT_PATH).is_file() and (self.repo_dir / "sam2" / "configs" / "sam2.1" / SAM2_CONFIG_NAME).is_file():
            return self.repo_dir
        env_value = os.environ.get("SAM2_REPO_DIR", "").strip()
        candidates = []
        if env_value:
            candidate = Path(env_value)
            if not candidate.is_absolute():
                candidate = Path(__file__).resolve().parents[1] / candidate
            candidates.append(candidate)
        base_dir = Path(__file__).resolve().parents[1]
        candidates.extend([base_dir, Path(__file__).resolve().parent, base_dir / "model", base_dir / "models", base_dir.parent, base_dir.parent / "sam2", base_dir.parent / "model", base_dir.parent / "models"])
        for candidate in candidates:
            candidate = candidate.resolve(strict=False)
            if (candidate / SAM2_CHECKPOINT_PATH).is_file() and (candidate / "sam2" / "configs" / "sam2.1" / SAM2_CONFIG_NAME).is_file():
                self.repo_dir = candidate
                return candidate
        for root in candidates:
            root = root.resolve(strict=False)
            if not root.exists():
                continue
            try:
                for checkpoint in root.rglob(SAM2_CHECKPOINT_NAME):
                    parent = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
                    if (parent / "sam2" / "configs" / "sam2.1" / SAM2_CONFIG_NAME).is_file():
                        self.repo_dir = parent
                        return parent
            except OSError:
                continue
        raise FileNotFoundError("SAM 2 repository not found. Set SAM2_REPO_DIR to the folder containing sam2.1_hiera_small.pt and sam2/configs/sam2.1/sam2.1_hiera_s.yaml.")

    def load(self):
        if self.model is not None:
            return
        if self.device != "cuda":
            raise RuntimeError("CUDA GPU is required for SAM 2.")
        repo_dir = self.find_repo_dir()
        checkpoint = repo_dir / SAM2_CHECKPOINT_PATH
        config = repo_dir / "sam2" / "configs" / "sam2.1" / SAM2_CONFIG_NAME
        print("Loading SAM 2 Small...")
        self.model = build_sam2(str(config), str(checkpoint), device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        print("SAM 2 Small loaded.")

    def load_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError("Image must be a file path or PIL Image.")
        self.image = image
        self.image_np = np.array(image)
        self.current_mask = None
        self.accumulated_mask = np.zeros((image.height, image.width), dtype=bool)
        self.prompt_points = []
        self.prompt_labels = []
        if self.predictor is None:
            self.load()
        self.predictor.set_image(self.image_np)

    def set_image(self, image):
        self.load_image(image)

    def add_point(self, x, y, positive=True):
        if self.image is None:
            raise RuntimeError("No image has been loaded.")
        x = int(x)
        y = int(y)
        if x < 0 or y < 0 or x >= self.image.width or y >= self.image.height:
            return
        self.prompt_points.append((x, y))
        self.prompt_labels.append(1 if positive else 0)

    def remove_last_point(self):
        if self.prompt_points:
            self.prompt_points.pop()
            self.prompt_labels.pop()

    def clear_points(self):
        self.prompt_points.clear()
        self.prompt_labels.clear()
        self.current_mask = None

    def predict(self):
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
        best_index = int(np.argmax(scores))
        self.current_mask = masks[best_index].astype(bool)
        return self.current_mask, float(scores[best_index])

    def append_segment(self):
        if self.current_mask is None:
            raise RuntimeError("No current segment is available.")
        if self.accumulated_mask is None:
            self.accumulated_mask = np.zeros_like(self.current_mask, dtype=bool)
        self.accumulated_mask |= self.current_mask
        appended_mask = self.current_mask.copy()
        self.current_mask = None
        self.clear_points()
        return appended_mask

    def append_point_segment(self, x, y, positive=True):
        self.add_point(x, y, positive)
        self.predict()
        self.append_segment()
        return self.get_accumulated_mask()

    def build_inpainting_mask(self, target_size=None, crop_box=None, original_size=None):
        accumulated = self.get_accumulated_mask()
        if accumulated is None or not np.any(accumulated):
            raise RuntimeError("No accumulated selection is available.")
        raw_mask = Image.fromarray((accumulated.astype(np.uint8) * 255), mode="L")
        if original_size is not None and crop_box is not None:
            width, height = original_size
            left, top, right, bottom = [float(value) for value in crop_box]
            x1 = max(0, min(width - 1, int(round(left * width))))
            y1 = max(0, min(height - 1, int(round(top * height))))
            x2 = max(x1 + 1, min(width, int(round(right * width))))
            y2 = max(y1 + 1, min(height, int(round(bottom * height))))
            raw_mask = raw_mask.crop((x1, y1, x2, y2))
        if target_size is not None and raw_mask.size != tuple(target_size):
            raw_mask = raw_mask.resize(tuple(target_size), Image.Resampling.NEAREST)
        return raw_mask.copy()

    def select_point(self, x, y, positive=True, target_size=None, crop_box=None, original_size=None):
        self.append_point_segment(x, y, positive)
        return self.build_inpainting_mask(target_size=target_size, crop_box=crop_box, original_size=original_size)

    def clear_all_segments(self):
        self.clear_selection()
        return self.get_accumulated_mask()

    def clear_selection(self):
        if self.image is None:
            self.accumulated_mask = None
        else:
            self.accumulated_mask = np.zeros((self.image.height, self.image.width), dtype=bool)
        self.current_mask = None
        self.clear_points()

    def get_current_mask(self):
        return None if self.current_mask is None else self.current_mask.copy()

    def get_accumulated_mask(self):
        return None if self.accumulated_mask is None else self.accumulated_mask.copy()

    def get_prompt_points(self):
        return list(self.prompt_points), list(self.prompt_labels)

    def get_device(self):
        return self.device

    def get_image_size(self):
        if self.image is None:
            return None
        return self.image.size

    def save_selection(self, path):
        if self.image is None:
            raise RuntimeError("No image has been loaded.")
        if self.accumulated_mask is None or not np.any(self.accumulated_mask):
            raise RuntimeError("No accumulated selection is available.")
        source = np.array(self.image.convert("RGBA"))
        source[:, :, 3] = self.accumulated_mask.astype(np.uint8) * 255
        Image.fromarray(source, mode="RGBA").save(path)

    def get_gpu_name(self):
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)

    def close(self):
        try:
            if self.predictor is not None:
                self.predictor.reset_predictor()
        except Exception:
            pass
        self.predictor = None
        self.model = None
        self.image = None
        self.image_np = None
        self.current_mask = None
        self.accumulated_mask = None
        self.prompt_points = []
        self.prompt_labels = []
        self.clear_cache()

    def clear_cache(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
