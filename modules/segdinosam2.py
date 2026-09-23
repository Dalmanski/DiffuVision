import os
import sys
import gc
import socket
import re
from io import BytesIO
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoImageProcessor, AutoModelForSemanticSegmentation
from rembg import new_session, remove

def _load_hf_model(loader, model_name, **kwargs):
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}
    if not offline:
        try:
            socket.create_connection(("huggingface.co", 443), timeout=1)
        except OSError:
            offline = True
    kwargs.setdefault("local_files_only", offline)
    return loader.from_pretrained(model_name, **kwargs)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_REPO_DIR = os.path.join(SCRIPT_DIR, "sam2_repo")
SAM2_CHECKPOINT_NAME = "sam2.1_hiera_tiny.pt"
SAM2_CHECKPOINT_PATH = os.path.join("checkpoints", SAM2_CHECKPOINT_NAME)
SAM2_CONFIG_NAME = "sam2.1_hiera_t.yaml"
if SAM2_REPO_DIR not in sys.path:
    sys.path.insert(0, SAM2_REPO_DIR)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
REMBG_MODEL_ID = "u2net"
SEGFORMER_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
FASHN_MODEL_ID = "fashn-ai/fashn-human-parser"
DINO_PREFERRED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM2_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEGFORMER_PREFERRED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FASHN_PREFERRED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REMBG_SESSION_PROVIDERS = ["CPUExecutionProvider"]
DINO_BOX_THRESHOLD = 0.30
DINO_TEXT_THRESHOLD = 0.20
NEGATIVE_BOX_THRESHOLD = 0.18
NEGATIVE_TEXT_THRESHOLD = 0.15
MAX_CLOTHING_BOX_AREA_RATIO = 0.60
MAX_NEGATIVE_BOX_AREA_RATIO = 0.45
MAX_PERSON_BOX_AREA_RATIO = 0.98
SUSPICIOUS_NEGATIVE_HEIGHT_RATIO = 0.75
SUSPICIOUS_NEGATIVE_WIDTH_RATIO = 0.30
SUSPICIOUS_NEGATIVE_AREA_RATIO = 0.25
NEGATIVE_MASK_MAX_POSITIVE_OVERLAP_RATIO = 0.92
MIN_BOX_AREA_RATIO = 0.002
DUPLICATE_IOU_THRESHOLD = 0.55
NESTED_CONTAINMENT_THRESHOLD = 0.90
MAX_CLOTHING_DETECTIONS = 2
MAX_NEGATIVE_DETECTIONS = 2
MAX_PERSON_DETECTIONS = 8
MIN_CLEANED_MASK_AREA = 100
MIN_REMAINING_MASK_RATIO = 0.10
REMBG_MASK_THRESHOLD = 8
REMBG_MASK_ERODE_PIXELS = 1
COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW", "MAGENTA", "CYAN", "PURPLE", "ORANGE", "SKY BLUE", "LIME"]
SEGMENT_COLORS = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0), (0, 128, 255), (128, 255, 0)], dtype=np.uint8)
SEGFORMER_CLASS_IDS = {"background": [0], "hat": [1], "hair": [2], "sunglasses": [3], "upper-clothes": [4], "skirt": [5], "pants": [6], "dress": [7], "belt": [8], "left-shoe": [9], "right-shoe": [10], "face": [11], "left-leg": [12], "right-leg": [13], "left-arm": [14], "right-arm": [15], "bag": [16], "scarf": [17]}
FASHN_CLASS_IDS = {"background": [0], "face": [1], "hair": [2], "top": [3], "dress": [4], "skirt": [5], "pants": [6], "belt": [7], "bag": [8], "hat": [9], "scarf": [10], "glasses": [11], "arms": [12], "hands": [13], "legs": [14], "feet": [15], "torso": [16], "jewelry": [17]}
AVAILABLE_MODELS = {"dino", "rembg", "segformer_b2", "fashn"}

@dataclass
class PromptInstruction:
    model: str
    text: str

@dataclass
class SegmentationResult:
    base_masks: list
    masks: list
    labels: list
    legend: list
    colored_result: Image.Image

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def adjust_mask_thickness(mask, thickness):
    mask = np.asarray(mask).astype(bool)
    thickness = int(thickness)
    if thickness == 0:
        return mask.copy()
    result = mask.copy()
    steps = abs(thickness)
    if thickness > 0:
        for _ in range(steps):
            padded = np.pad(result, 1, mode="constant", constant_values=False)
            result = padded[0:-2, 0:-2] | padded[0:-2, 1:-1] | padded[0:-2, 2:] | padded[1:-1, 0:-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:] | padded[2:, 0:-2] | padded[2:, 1:-1] | padded[2:, 2:]
    else:
        for _ in range(steps):
            padded = np.pad(result, 1, mode="constant", constant_values=False)
            result = padded[0:-2, 0:-2] & padded[0:-2, 1:-1] & padded[0:-2, 2:] & padded[1:-1, 0:-2] & padded[1:-1, 1:-1] & padded[1:-1, 2:] & padded[2:, 0:-2] & padded[2:, 1:-1] & padded[2:, 2:]
    return result.astype(bool)

def create_colored_result(image, masks, labels):
    image_np = np.asarray(image).copy()
    result = image_np.copy()
    legend = []
    for index, (mask, label) in enumerate(zip(masks, labels)):
        pixels = np.asarray(mask).astype(bool)
        if not np.any(pixels):
            continue
        color_index = index % len(SEGMENT_COLORS)
        color = SEGMENT_COLORS[color_index]
        color_name = COLOR_NAMES[color_index]
        original = result[pixels].astype(np.float32)
        color_float = color.astype(np.float32)
        blended = original * 0.50 + color_float * 0.50
        result[pixels] = np.clip(blended, 0, 255).astype(np.uint8)
        legend.append((color_name, color.copy(), label))
    return Image.fromarray(result), legend

class RembgSegmenter:
    def __init__(self):
        self.session = None

    def load(self):
        if self.session is not None:
            return
        print()
        print("Loading Rembg...")
        self.session = new_session(REMBG_MODEL_ID, providers=REMBG_SESSION_PROVIDERS)
        print("Rembg loaded.")

    def unload(self):
        self.session = None
        gc.collect()
        clear_cuda()

    def segment_foreground(self, image):
        if self.session is None:
            self.load()
        image = image.convert("RGBA")
        output = remove(image, session=self.session, only_mask=True)
        if isinstance(output, Image.Image):
            mask_image = output.convert("L")
        elif isinstance(output, (bytes, bytearray, memoryview)):
            mask_image = Image.open(BytesIO(bytes(output))).convert("L")
        elif isinstance(output, np.ndarray):
            array = np.asarray(output)
            if array.ndim > 2:
                array = np.squeeze(array)
            if array.ndim != 2:
                raise TypeError(f"Unsupported rembg mask array shape: {array.shape}")
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
                    array = np.clip(array * 255.0, 0, 255)
                array = np.clip(array, 0, 255).astype(np.uint8)
            mask_image = Image.fromarray(array, "L")
        else:
            try:
                mask_image = Image.fromarray(np.asarray(output, dtype=np.uint8), "L")
            except Exception as e:
                raise TypeError(f"Unsupported rembg output type: {type(output).__name__}") from e
        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
        mask_array = np.asarray(mask_image, dtype=np.uint8)
        mask = mask_array > REMBG_MASK_THRESHOLD
        if REMBG_MASK_ERODE_PIXELS > 0:
            mask = adjust_mask_thickness(mask, -REMBG_MASK_ERODE_PIXELS)
        return mask.astype(bool)

    def segment_background(self, image):
        foreground = self.segment_foreground(image)
        return (~foreground).astype(bool)

class GroundingDINO:
    def __init__(self):
        self.processor = None
        self.model = None
        self.preferred_device = DINO_PREFERRED_DEVICE
        self.runtime_device = DINO_PREFERRED_DEVICE

    def load(self):
        if self.model is not None:
            return
        print()
        print("Loading Grounding DINO...")
        self.processor = _load_hf_model(AutoProcessor, GROUNDING_DINO_MODEL_ID, use_fast=False)
        print("  > Processor loaded.")
        if self.preferred_device == "cuda":
            try:
                print("  > Loading DINO on GPU...")
                self.model = _load_hf_model(AutoModelForZeroShotObjectDetection, GROUNDING_DINO_MODEL_ID)
                self.model = self.model.to("cuda")
                self.model.eval()
                self.runtime_device = "cuda"
                print("  > DINO GPU ready.")
                self.print_cuda_memory()
            except Exception as e:
                print("  > DINO GPU load failed.")
                print(f"  > {e}")
                self.unload()
                self.processor = _load_hf_model(AutoProcessor, GROUNDING_DINO_MODEL_ID, use_fast=False)
                self.model = _load_hf_model(AutoModelForZeroShotObjectDetection, GROUNDING_DINO_MODEL_ID)
                self.model = self.model.to("cpu")
                self.model.eval()
                self.runtime_device = "cpu"
                print("  > DINO CPU fallback ready.")
        else:
            self.model = _load_hf_model(AutoModelForZeroShotObjectDetection, GROUNDING_DINO_MODEL_ID)
            self.model = self.model.to("cpu")
            self.model.eval()
            self.runtime_device = "cpu"
            print("  > DINO CPU ready.")

    def unload(self):
        if self.model is not None:
            try:
                self.model = self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        self.processor = None
        gc.collect()
        clear_cuda()

    def print_cuda_memory(self):
        if not torch.cuda.is_available():
            return
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = total_bytes - free_bytes
            print(f"  CUDA Used: {used_bytes / 1024**2:.0f} MB")
            print(f"  CUDA Free: {free_bytes / 1024**2:.0f} MB")
            print(f"  CUDA Total: {total_bytes / 1024**2:.0f} MB")
        except Exception:
            pass

    def switch_to_cpu(self):
        if self.model is None:
            return
        try:
            self.model = self.model.to("cpu")
            self.model.eval()
        except Exception:
            pass
        self.runtime_device = "cpu"
        clear_cuda()

    def detect(self, image, categories, box_threshold, text_threshold, max_area_ratio, mode):
        if self.model is None or self.processor is None:
            raise RuntimeError("Grounding DINO is not loaded.")
        if not categories:
            return [], [], []
        all_boxes = []
        all_labels = []
        all_scores = []
        if mode == "clothing":
            max_detections = MAX_CLOTHING_DETECTIONS
        elif mode == "person":
            max_detections = MAX_PERSON_DETECTIONS
        else:
            max_detections = MAX_NEGATIVE_DETECTIONS
        for category in categories:
            print()
            print("-" * 70)
            print(f"{mode.upper()} QUERY: {category}")
            inputs = self.processor(images=image, text=[[category]], return_tensors="pt")
            for key, value in list(inputs.items()):
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(self.runtime_device)
            try:
                with torch.inference_mode():
                    outputs = self.model(**inputs)
            except RuntimeError as error:
                text = str(error).lower()
                gpu_problem = self.runtime_device == "cuda" and ("out of memory" in text or "expected scalar type" in text or "mat1 and mat2" in text)
                if gpu_problem:
                    print("  DINO GPU issue detected.")
                    print("  Switching to CPU.")
                    self.switch_to_cpu()
                    inputs = self.processor(images=image, text=[[category]], return_tensors="pt")
                    for key, value in list(inputs.items()):
                        if isinstance(value, torch.Tensor):
                            inputs[key] = value.to("cpu")
                    with torch.inference_mode():
                        outputs = self.model(**inputs)
                else:
                    raise
            try:
                results = self.processor.post_process_grounded_object_detection(outputs, inputs["input_ids"], threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(image.height, image.width)])
            except TypeError:
                results = self.processor.post_process_grounded_object_detection(outputs=outputs, input_ids=inputs["input_ids"], threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(image.height, image.width)])
            result = results[0]
            raw_boxes = result.get("boxes", torch.empty((0, 4)))
            raw_scores = result.get("scores", torch.empty(0))
            boxes = raw_boxes.detach().float().cpu().numpy()
            scores = raw_scores.detach().float().cpu().numpy()
            local = []
            image_area = image.width * image.height
            for box, score in zip(boxes, scores):
                if len(box) != 4:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                x1 = np.clip(x1, 0, image.width - 1)
                y1 = np.clip(y1, 0, image.height - 1)
                x2 = np.clip(x2, 0, image.width - 1)
                y2 = np.clip(y2, 0, image.height - 1)
                width = x2 - x1
                height = y2 - y1
                if width <= 1 or height <= 1:
                    continue
                area_ratio = width * height / image_area
                if area_ratio > max_area_ratio:
                    print(f"  Rejected {category}: area={area_ratio:.1%}")
                    continue
                if area_ratio < MIN_BOX_AREA_RATIO:
                    continue
                if mode == "negative":
                    width_ratio = width / image.width
                    height_ratio = height / image.height
                    person_like = height_ratio >= SUSPICIOUS_NEGATIVE_HEIGHT_RATIO and width_ratio >= SUSPICIOUS_NEGATIVE_WIDTH_RATIO
                    very_large = area_ratio >= SUSPICIOUS_NEGATIVE_AREA_RATIO
                    if person_like and very_large:
                        print(f"  Rejected {category}: suspicious person-sized negative box={box.astype(int)} area={area_ratio:.1%}")
                        continue
                local.append((np.array([x1, y1, x2, y2], dtype=np.float32), float(score)))
            local.sort(key=lambda item: item[1], reverse=True)
            local = local[:max_detections]
            for box, score in local:
                all_boxes.append(box)
                all_labels.append(category)
                all_scores.append(score)
                print(f"  Candidate: {category} score={score:.3f} box={box.astype(int)}")
            del inputs
            del outputs
            del results
            del result
            gc.collect()
            if self.runtime_device == "cuda":
                clear_cuda()
        boxes, labels, scores = self.remove_duplicate_boxes(all_boxes, all_labels, all_scores)
        boxes, labels, scores = self.remove_nested_boxes(boxes, labels, scores)
        return boxes, labels, scores

    def remove_duplicate_boxes(self, boxes, labels, scores):
        if len(boxes) <= 1:
            return boxes, labels, scores
        order = np.argsort(-np.asarray(scores))
        kept_boxes = []
        kept_labels = []
        kept_scores = []
        for index in order:
            index = int(index)
            box = boxes[index]
            label = labels[index]
            score = scores[index]
            duplicate = False
            for old_index, old_box in enumerate(kept_boxes):
                if label != kept_labels[old_index]:
                    continue
                iou = self.box_iou(box, old_box)
                if iou >= DUPLICATE_IOU_THRESHOLD:
                    duplicate = True
                    print(f"  Duplicate removed: {label} IoU={iou:.2f}")
                    break
            if not duplicate:
                kept_boxes.append(box)
                kept_labels.append(label)
                kept_scores.append(score)
        return kept_boxes, kept_labels, kept_scores

    def remove_nested_boxes(self, boxes, labels, scores):
        if len(boxes) <= 1:
            return boxes, labels, scores
        keep = [True for _ in boxes]
        areas = []
        for box in boxes:
            x1, y1, x2, y2 = box
            areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        for i in range(len(boxes)):
            if not keep[i]:
                continue
            for j in range(len(boxes)):
                if i == j:
                    continue
                if not keep[j]:
                    continue
                if labels[i] != labels[j]:
                    continue
                containment = self.box_containment(boxes[i], boxes[j])
                if containment >= NESTED_CONTAINMENT_THRESHOLD and areas[j] > areas[i]:
                    keep[i] = False
                    print(f"  Nested box removed: {labels[i]}")
                    break
        filtered_boxes = []
        filtered_labels = []
        filtered_scores = []
        for index in range(len(boxes)):
            if keep[index]:
                filtered_boxes.append(boxes[index])
                filtered_labels.append(labels[index])
                filtered_scores.append(scores[index])
        return filtered_boxes, filtered_labels, filtered_scores

    @staticmethod
    def box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        intersection = width * height
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    @staticmethod
    def box_containment(outer, inner):
        ox1, oy1, ox2, oy2 = outer
        ix1, iy1, ix2, iy2 = inner
        x1 = max(ox1, ix1)
        y1 = max(oy1, iy1)
        x2 = min(ox2, ix2)
        y2 = min(oy2, iy2)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        intersection = width * height
        inner_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inner_area <= 0:
            return 0.0
        return intersection / inner_area

class SAM2:
    def __init__(self, repo_dir=SAM2_REPO_DIR, device=SAM2_DEVICE):
        self.repo_dir = repo_dir
        self.device = device
        self.model = None
        self.predictor = None

    def load(self):
        if self.model is not None:
            return
        checkpoint = os.path.join(self.repo_dir, SAM2_CHECKPOINT_PATH)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError("SAM 2 checkpoint not found:\n\n" + checkpoint)
        config = os.path.join(self.repo_dir, "sam2", "configs", "sam2.1", SAM2_CONFIG_NAME)
        if not os.path.isfile(config):
            raise FileNotFoundError("SAM 2 config not found:\n\n" + config)
        print(f"Loading SAM 2 ({SAM2_CHECKPOINT_NAME}, {SAM2_CONFIG_NAME})...")
        self.model = build_sam2(config, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        print(f"SAM 2 ({SAM2_CHECKPOINT_NAME}, {SAM2_CONFIG_NAME}) loaded.")

    def unload(self):
        self.predictor = None
        self.model = None
        gc.collect()
        clear_cuda()

    def set_image(self, image):
        if self.predictor is None:
            raise RuntimeError("SAM 2 is not loaded.")
        self.predictor.set_image(image)

    def predict(self, box):
        if self.predictor is None:
            raise RuntimeError("SAM 2 is not loaded.")
        masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM 2 returned no masks.")
        scores = np.asarray(scores).reshape(-1)
        best_index = int(np.argmax(scores))
        mask = np.asarray(masks[best_index])
        mask = np.squeeze(mask)
        return mask.astype(bool), float(scores[best_index])

    @staticmethod
    def clip_mask(mask, box, shape):
        height, width = shape
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        result = np.zeros((height, width), dtype=bool)
        if x2 <= x1 or y2 <= y1:
            return result
        result[y1:y2 + 1, x1:x2 + 1] = mask[y1:y2 + 1, x1:x2 + 1] > 0
        return result

class SegformerB2Clothes:
    def __init__(self):
        self.processor = None
        self.model = None
        self.preferred_device = SEGFORMER_PREFERRED_DEVICE
        self.runtime_device = SEGFORMER_PREFERRED_DEVICE

    def load(self):
        if self.model is not None:
            return
        print()
        print("Loading SegFormer B2 Clothes...")
        self.processor = _load_hf_model(AutoImageProcessor, SEGFORMER_MODEL_ID, use_fast=False)
        if self.preferred_device == "cuda":
            try:
                print("  > Loading SegFormer on GPU...")
                self.model = _load_hf_model(AutoModelForSemanticSegmentation, SEGFORMER_MODEL_ID)
                self.model = self.model.to("cuda")
                self.model.eval()
                self.runtime_device = "cuda"
                print("  > SegFormer GPU ready.")
            except Exception as e:
                print("  > SegFormer GPU load failed.")
                print(f"  > {e}")
                self.unload()
                self.processor = _load_hf_model(AutoImageProcessor, SEGFORMER_MODEL_ID, use_fast=False)
                self.model = _load_hf_model(AutoModelForSemanticSegmentation, SEGFORMER_MODEL_ID)
                self.model = self.model.to("cpu")
                self.model.eval()
                self.runtime_device = "cpu"
                print("  > SegFormer CPU fallback ready.")
        else:
            self.model = _load_hf_model(AutoModelForSemanticSegmentation, SEGFORMER_MODEL_ID)
            self.model = self.model.to("cpu")
            self.model.eval()
            self.runtime_device = "cpu"
            print("  > SegFormer CPU ready.")

    def unload(self):
        if self.model is not None:
            try:
                self.model = self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        self.processor = None
        gc.collect()
        clear_cuda()

    def resolve_class_ids(self, category):
        normalized = str(category).strip().lower()
        if normalized not in SEGFORMER_CLASS_IDS:
            supported = ", ".join(SEGFORMER_CLASS_IDS.keys())
            raise ValueError(f"SegFormer category '{category}' is not supported. Supported labels: {supported}")
        return SEGFORMER_CLASS_IDS[normalized], normalized

    def predict(self, image, categories):
        if self.model is None or self.processor is None:
            raise RuntimeError("SegFormer B2 Clothes is not loaded.")
        masks = []
        labels = []
        inputs = self.processor(images=image, return_tensors="pt")
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.runtime_device)
        try:
            with torch.inference_mode():
                outputs = self.model(**inputs)
        except RuntimeError as error:
            text = str(error).lower()
            gpu_problem = self.runtime_device == "cuda" and ("out of memory" in text or "expected scalar type" in text or "mat1 and mat2" in text)
            if not gpu_problem:
                raise
            print("  SegFormer GPU issue detected.")
            print("  Switching to CPU.")
            self.model = self.model.to("cpu")
            self.runtime_device = "cpu"
            inputs = self.processor(images=image, return_tensors="pt")
            for key, value in list(inputs.items()):
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to("cpu")
            with torch.inference_mode():
                outputs = self.model(**inputs)
        logits = outputs.logits
        logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
        prediction = logits.argmax(dim=1)[0].detach().cpu().numpy()
        for category in categories:
            class_ids, normalized = self.resolve_class_ids(category)
            mask = np.isin(prediction, np.asarray(class_ids, dtype=np.int64))
            if np.any(mask):
                masks.append(mask.astype(bool))
                labels.append(category)
                print(f"  SegFormer: {category} -> {normalized} pixels={int(np.count_nonzero(mask))}")
            else:
                print(f"  SegFormer: {category} produced an empty mask.")
        del inputs
        del outputs
        del logits
        del prediction
        gc.collect()
        if self.runtime_device == "cuda":
            clear_cuda()
        return masks, labels

class FashnHumanParser:
    def __init__(self):
        self.processor = None
        self.model = None
        self.preferred_device = FASHN_PREFERRED_DEVICE
        self.runtime_device = FASHN_PREFERRED_DEVICE

    def load(self):
        if self.model is not None:
            return
        print()
        print("Loading FASHN Human Parser...")
        self.processor = _load_hf_model(AutoImageProcessor, FASHN_MODEL_ID, use_fast=False)
        if self.preferred_device == "cuda":
            try:
                print("  > Loading FASHN on GPU...")
                self.model = _load_hf_model(AutoModelForSemanticSegmentation, FASHN_MODEL_ID)
                self.model = self.model.to("cuda")
                self.model.eval()
                self.runtime_device = "cuda"
                print("  > FASHN GPU ready.")
            except Exception as e:
                print("  > FASHN GPU load failed.")
                print(f"  > {e}")
                self.unload()
                self.processor = _load_hf_model(AutoImageProcessor, FASHN_MODEL_ID, use_fast=False)
                self.model = _load_hf_model(AutoModelForSemanticSegmentation, FASHN_MODEL_ID)
                self.model = self.model.to("cpu")
                self.model.eval()
                self.runtime_device = "cpu"
                print("  > FASHN CPU fallback ready.")
        else:
            self.model = _load_hf_model(AutoModelForSemanticSegmentation, FASHN_MODEL_ID)
            self.model = self.model.to("cpu")
            self.model.eval()
            self.runtime_device = "cpu"
            print("  > FASHN CPU ready.")

    def unload(self):
        if self.model is not None:
            try:
                self.model = self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        self.processor = None
        gc.collect()
        clear_cuda()

    def resolve_class_ids(self, category):
        normalized = str(category).strip().lower()
        if normalized not in FASHN_CLASS_IDS:
            supported = ", ".join(FASHN_CLASS_IDS.keys())
            raise ValueError(f"FASHN category '{category}' is not supported. Supported labels: {supported}")
        return FASHN_CLASS_IDS[normalized], normalized

    def predict(self, image, categories):
        if self.model is None or self.processor is None:
            raise RuntimeError("FASHN Human Parser is not loaded.")
        masks = []
        labels = []
        inputs = self.processor(images=image, return_tensors="pt")
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.runtime_device)
        try:
            with torch.inference_mode():
                outputs = self.model(**inputs)
        except RuntimeError as error:
            text = str(error).lower()
            gpu_problem = self.runtime_device == "cuda" and ("out of memory" in text or "expected scalar type" in text or "mat1 and mat2" in text)
            if not gpu_problem:
                raise
            print("  FASHN GPU issue detected.")
            print("  Switching to CPU.")
            self.model = self.model.to("cpu")
            self.runtime_device = "cpu"
            inputs = self.processor(images=image, return_tensors="pt")
            for key, value in list(inputs.items()):
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to("cpu")
            with torch.inference_mode():
                outputs = self.model(**inputs)
        logits = outputs.logits
        logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
        prediction = logits.argmax(dim=1)[0].detach().cpu().numpy()
        for category in categories:
            class_ids, normalized = self.resolve_class_ids(category)
            mask = np.isin(prediction, np.asarray(class_ids, dtype=np.int64))
            if np.any(mask):
                masks.append(mask.astype(bool))
                labels.append(category)
                print(f"  FASHN: {category} -> {normalized} pixels={int(np.count_nonzero(mask))}")
            else:
                print(f"  FASHN: {category} produced an empty mask.")
        del inputs
        del outputs
        del logits
        del prediction
        gc.collect()
        if self.runtime_device == "cuda":
            clear_cuda()
        return masks, labels

class SegDinoSAM2:
    def __init__(self):
        self.preferred_dino_device = DINO_PREFERRED_DEVICE
        self.sam2_device = SAM2_DEVICE
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            try:
                self.gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                self.gpu_name = "CUDA"
        else:
            self.gpu_name = "CPU"
        self.dino = GroundingDINO()
        self.sam2 = SAM2()
        self.rembg = RembgSegmenter()
        self.segformer_b2 = SegformerB2Clothes()
        self.fashn = FashnHumanParser()

    def load_dino(self):
        self.dino.load()

    def unload_dino(self):
        self.dino.unload()

    def load_sam2(self):
        self.sam2.load()

    def unload_sam2(self):
        self.sam2.unload()

    def load_rembg(self):
        self.rembg.load()

    def unload_rembg(self):
        self.rembg.unload()

    def load_segformer_b2(self):
        self.segformer_b2.load()

    def unload_segformer_b2(self):
        self.segformer_b2.unload()

    def load_fashn(self):
        self.fashn.load()

    def unload_fashn(self):
        self.fashn.unload()

    def normalize_model_name(self, model_name):
        model_name = str(model_name).strip().lower()
        if model_name not in AVAILABLE_MODELS:
            supported = ", ".join(AVAILABLE_MODELS)
            raise ValueError(f"Unknown model '@{model_name}:'. Supported model names: {supported}")
        return model_name

    def parse_prompt(self, text):
        if text is None:
            return []
        text = str(text).strip()
        if not text:
            return []
        text = text.replace("\n", " ")
        instructions = []
        pattern = re.compile(r"@([A-Za-z0-9_]+)\s*:\s*\[([^\]]*)\]|@([A-Za-z0-9_]+)\s*:\s*([^,@\[]+)|([^,@]+)")
        for match in pattern.finditer(text):
            grouped_model = match.group(1)
            grouped_text = match.group(2)
            single_model = match.group(3)
            single_text = match.group(4)
            default_text = match.group(5)
            if grouped_model is not None:
                model = self.normalize_model_name(grouped_model)
                categories = [item.strip() for item in grouped_text.split(",") if item.strip()]
                if not categories:
                    raise ValueError(f"Empty model group for @{grouped_model}:")
                for category in categories:
                    instructions.append(PromptInstruction(model=model, text=category))
                continue
            if single_model is not None:
                model = self.normalize_model_name(single_model)
                category = single_text.strip()
                if not category:
                    raise ValueError(f"Empty category for @{single_model}:")
                instructions.append(PromptInstruction(model=model, text=category))
                continue
            if default_text is not None:
                category = default_text.strip()
                if category:
                    instructions.append(PromptInstruction(model="dino", text=category))
        return instructions

    def split_instructions(self, instructions):
        groups = {model: [] for model in AVAILABLE_MODELS}
        for instruction in instructions:
            groups[instruction.model].append(instruction.text)
        return groups

    def segment_rembg_instruction(self, image, category):
        normalized = str(category).strip().lower()
        if normalized == "person":
            return self.rembg.segment_foreground(image), "person"
        if normalized == "background":
            return self.rembg.segment_background(image), "background"
        raise ValueError(f"@rembg: only 'person' and 'background' are supported, received '{category}'.")

    def segment(self, image, positive_prompt, negative_prompt, thickness=0):
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image.")
        image = image.convert("RGB")
        image_np = np.array(image)
        positive_instructions = self.parse_prompt(positive_prompt)
        negative_instructions = self.parse_prompt(negative_prompt)
        positive_groups = self.split_instructions(positive_instructions)
        negative_groups = self.split_instructions(negative_instructions)
        if not positive_instructions:
            positive_groups["dino"] = ["all"]
        positive_boxes = []
        positive_labels = []
        positive_scores = []
        negative_boxes = []
        negative_labels = []
        negative_scores = []
        positive_masks = []
        positive_mask_labels = []
        negative_masks = []
        negative_mask_labels = []
        if positive_groups["dino"]:
            self._ensure_dino()
            positive_boxes, positive_labels, positive_scores = self.dino.detect(image=image, categories=positive_groups["dino"], box_threshold=DINO_BOX_THRESHOLD, text_threshold=DINO_TEXT_THRESHOLD, max_area_ratio=MAX_CLOTHING_BOX_AREA_RATIO, mode="clothing")
            if len(positive_boxes) == 0:
                self.unload_dino()
                raise RuntimeError("No positive object was detected by Grounding DINO.")
        if negative_groups["dino"]:
            self._ensure_dino()
            negative_boxes, negative_labels, negative_scores = self.dino.detect(image=image, categories=negative_groups["dino"], box_threshold=NEGATIVE_BOX_THRESHOLD, text_threshold=NEGATIVE_TEXT_THRESHOLD, max_area_ratio=MAX_NEGATIVE_BOX_AREA_RATIO, mode="negative")
        if self.dino.model is not None:
            self.unload_dino()
        if positive_groups["rembg"]:
            self._ensure_rembg()
            for category in positive_groups["rembg"]:
                mask, label = self.segment_rembg_instruction(image, category)
                positive_masks.append(mask.astype(bool))
                positive_mask_labels.append(label)
        if negative_groups["rembg"]:
            self._ensure_rembg()
            for category in negative_groups["rembg"]:
                mask, label = self.segment_rembg_instruction(image, category)
                negative_masks.append(mask.astype(bool))
                negative_mask_labels.append(label)
        if positive_groups["segformer_b2"]:
            self._ensure_segformer_b2()
            masks, labels = self.segformer_b2.predict(image, positive_groups["segformer_b2"])
            positive_masks.extend(masks)
            positive_mask_labels.extend(labels)
        if negative_groups["segformer_b2"]:
            self._ensure_segformer_b2()
            masks, labels = self.segformer_b2.predict(image, negative_groups["segformer_b2"])
            negative_masks.extend(masks)
            negative_mask_labels.extend(labels)
        if positive_groups["fashn"]:
            self._ensure_fashn()
            masks, labels = self.fashn.predict(image, positive_groups["fashn"])
            positive_masks.extend(masks)
            positive_mask_labels.extend(labels)
        if negative_groups["fashn"]:
            self._ensure_fashn()
            masks, labels = self.fashn.predict(image, negative_groups["fashn"])
            negative_masks.extend(masks)
            negative_mask_labels.extend(labels)
        if self.segformer_b2.model is not None:
            self.unload_segformer_b2()
        if self.fashn.model is not None:
            self.unload_fashn()
        if self.rembg.session is not None:
            self.unload_rembg()
        need_sam2 = len(positive_boxes) > 0 or len(negative_boxes) > 0
        if need_sam2:
            self._ensure_sam2()
            self.sam2.set_image(image_np)
        for box, label, score in zip(positive_boxes, positive_labels, positive_scores):
            mask, sam_score = self.sam2.predict(box)
            mask = self.sam2.clip_mask(mask, box, image_np.shape[:2])
            if np.any(mask):
                positive_masks.append(mask.astype(bool))
                positive_mask_labels.append(label)
        positive_union = np.zeros(image_np.shape[:2], dtype=bool)
        for mask in positive_masks:
            positive_union |= mask
        positive_union_area = np.count_nonzero(positive_union)
        for box, label, score in zip(negative_boxes, negative_labels, negative_scores):
            if self.sam2.model is None:
                self._ensure_sam2()
                self.sam2.set_image(image_np)
            mask, sam_score = self.sam2.predict(box)
            mask = self.sam2.clip_mask(mask, box, image_np.shape[:2]).astype(bool)
            mask_area = np.count_nonzero(mask)
            if mask_area == 0:
                continue
            overlap_area = np.count_nonzero(mask & positive_union)
            overlap_ratio = overlap_area / max(positive_union_area, 1)
            if positive_union_area > 0 and overlap_ratio >= NEGATIVE_MASK_MAX_POSITIVE_OVERLAP_RATIO and mask_area > positive_union_area:
                print(f"  Negative mask rejected: {label} positive-overlap={overlap_ratio:.1%} mask-area={mask_area} positive-area={positive_union_area}")
                continue
            negative_masks.append(mask)
            negative_mask_labels.append(label)
        if self.sam2.model is not None:
            self.unload_sam2()
        if not positive_masks:
            raise RuntimeError("No positive masks were produced.")
        negative_union = np.zeros(image_np.shape[:2], dtype=bool)
        for mask in negative_masks:
            negative_union |= mask
        cleaned_masks = []
        cleaned_labels = []
        occupied = np.zeros(image_np.shape[:2], dtype=bool)
        for mask, label in zip(positive_masks, positive_mask_labels):
            original_area = np.count_nonzero(mask)
            cleaned = mask & ~negative_union
            cleaned &= ~occupied
            cleaned_area = np.count_nonzero(cleaned)
            if cleaned_area < MIN_CLEANED_MASK_AREA:
                continue
            if original_area > 0 and cleaned_area < int(original_area * MIN_REMAINING_MASK_RATIO):
                continue
            cleaned_masks.append(cleaned.astype(bool))
            cleaned_labels.append(label)
            occupied |= cleaned
        if not cleaned_masks:
            raise RuntimeError("No usable masks remained.")
        base_masks = [mask.copy() for mask in cleaned_masks]
        adjusted_masks = [adjust_mask_thickness(mask, int(thickness)) for mask in base_masks]
        colored_result, legend = create_colored_result(image_np, adjusted_masks, cleaned_labels)
        return SegmentationResult(base_masks=base_masks, masks=adjusted_masks, labels=cleaned_labels, legend=legend, colored_result=colored_result)

    def _ensure_dino(self):
        if self.dino.model is None:
            self.dino.load()

    def _ensure_sam2(self):
        if self.sam2.model is None:
            self.sam2.load()

    def _ensure_rembg(self):
        if self.rembg.session is None:
            self.rembg.load()

    def _ensure_segformer_b2(self):
        if self.segformer_b2.model is None:
            self.segformer_b2.load()

    def _ensure_fashn(self):
        if self.fashn.model is None:
            self.fashn.load()