from PIL import Image, ImageFilter
import gc
import threading
import numpy as np
import torch
from modules.segdinosam2 import SegDinoSAM2, adjust_mask_thickness
from modules.selectsegsam2 import SAM2Segmenter
from modules.upscale_img import enhance
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SegmentImageMixin:

    def update_manual_segment_buttons(self):
        if not hasattr(self, 'manual_segment_btn'):
            return
        base_state = 'normal' if self.original_image is not None and not self.processing and not self.model_loading and not self.segmentation_loading and not self.classification_loading and not self.manual_segment_loading else 'disabled'
        self.manual_segment_btn.configure(state=base_state)
        self.clear_segment_btn.configure(state='normal' if self.original_image is not None else 'disabled')
        if self.manual_segment_mode:
            self.manual_segment_btn.configure(fg_color='#1f8f3a', hover_color='#176b2c')
        else:
            self.manual_segment_btn.configure(fg_color='#21262D', hover_color='#30363D')

    def toggle_manual_segment_mode(self):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        if self.original_image is None:
            return
        if self.manual_segment_mode:
            self.disable_manual_segment_mode()
            print('SAM2 point selection disabled')
            return
        try:
            self.sync_config()
        except Exception as e:
            print(f'Configuration Error: {e}')
            return
        self.manual_segment_mode = True
        self.manual_segment_loading = True
        self.manual_segment_btn.configure(state='disabled')
        self.clear_segment_btn.configure(state='normal' if self.original_image is not None else 'disabled')
        self.update_crop_button_state()
        print('Loading SAM2 point selection...')
        threading.Thread(target=self.load_manual_segmenter_worker, daemon=True).start()

    def load_manual_segmenter_worker(self):
        try:
            if self.sd_input_image is None:
                self.preprocess_uploaded_image(self.original_image)
            image = self.sd_input_image.copy() if self.sd_input_image is not None else self.input_image.copy()
            self.manual_segmenter = SAM2Segmenter()
            self.manual_segmenter.load_image(image)
            self.manual_selection_image = self.input_image.copy() if self.input_image is not None else self.original_image.copy()
            print('SAM2 point selection ready • click the image to append segments')
        except Exception as e:
            self.manual_segment_mode = False
            print(f'SAM2 point selection error: {e}')
            self.after(0, lambda err=str(e): print(f'SAM2 Error: {err}'))
        finally:
            self.manual_segment_loading = False
            self.after(0, self.update_crop_button_state)
            self.after(0, self.show_input)
            self.after(0, self.update_generate_state)

    def disable_manual_segment_mode(self):
        self.manual_segment_mode = False
        self.manual_segment_loading = False
        segmenter = self.manual_segmenter
        self.manual_segmenter = None
        self.manual_selection_image = None
        if segmenter is not None:
            segmenter.close()
        self.cleanup_gpu()
        self.reload_mask_btn.configure(state='normal' if self.original_image is not None and not self.processing and not self.segmentation_loading else 'disabled')
        self.update_crop_button_state()
        self.show_input()

    def select_manual_segment(self, event):
        if not self.manual_segment_mode or self.manual_segment_loading or self.manual_segmenter is None:
            return
        if self.input_display_info is None or self.manual_selection_image is None:
            return
        x, y, width, height = self.input_display_info
        if event.x < x or event.x >= x + width or event.y < y or event.y >= y + height:
            return
        image_width, image_height = self.manual_selection_image.size
        source_x = max(0, min(image_width - 1, int((event.x - x) / width * image_width)))
        source_y = max(0, min(image_height - 1, int((event.y - y) / height * image_height)))
        crop_left, crop_top, crop_right, crop_bottom = self.get_effective_crop_box()
        crop_x1 = max(0, min(image_width - 1, int(round(crop_left * image_width))))
        crop_y1 = max(0, min(image_height - 1, int(round(crop_top * image_height))))
        crop_x2 = max(crop_x1 + 1, min(image_width, int(round(crop_right * image_width))))
        crop_y2 = max(crop_y1 + 1, min(image_height, int(round(crop_bottom * image_height))))
        if not crop_x1 <= source_x < crop_x2 or not crop_y1 <= source_y < crop_y2:
            return
        segment_width, segment_height = self.manual_segmenter.get_image_size()
        image_x = min(segment_width - 1, int((source_x - crop_x1) / (crop_x2 - crop_x1) * segment_width))
        image_y = min(segment_height - 1, int((source_y - crop_y1) / (crop_y2 - crop_y1) * segment_height))
        self.manual_segment_loading = True
        self.update_manual_segment_buttons()
        threading.Thread(target=self.manual_segment_worker, args=(image_x, image_y), daemon=True).start()

    def manual_segment_worker(self, image_x, image_y):
        try:
            thickness = float(self.config.get('mask_thickness', 0.0))
            blur = float(self.config.get('mask_blur', 0.0))
            print(f'Manual mask settings: outline={thickness:g}px, blur={blur:g}px')
            mask = self.manual_segmenter.select_point(image_x, image_y, positive=True)
            mask = self.apply_mask_adjustments(mask, thickness, blur)
            existing_mask = self.mask_image
            if self.has_valid_mask(existing_mask):
                if existing_mask.size != mask.size:
                    existing_mask = existing_mask.resize(mask.size, Image.Resampling.NEAREST)
                combined = np.maximum(np.asarray(existing_mask, dtype=np.uint8), np.asarray(mask, dtype=np.uint8))
                mask = Image.fromarray(combined.astype(np.uint8), mode='L')
            self.mask_image = mask
            self.mask_source = 'manual'
            self.output_image = None
            self.output_showing_original = False
            self.after(0, self.show_input)
            self.after(0, self.show_output)
            print('Segment appended')
        except Exception as e:
            print(f'SAM2 selection error: {e}')
            self.after(0, lambda err=str(e): print(f'SAM2 Selection Error: {err}'))
        finally:
            self.manual_segment_loading = False
            self.after(0, self.update_crop_button_state)
            self.after(0, lambda: self.clear_segment_btn.configure(state='normal' if self.original_image is not None else 'disabled'))
            self.after(0, self.update_generate_state)

    def clear_manual_segments(self):
        if self.processing or self.manual_segment_loading:
            return
        if self.manual_segmenter is not None:
            self.manual_segmenter.clear_all_segments()
        self.reset_preview_state()
        self.generate_btn.configure(state='disabled')
        print('All manually appended segments removed')
        self.show_input()
        self.show_output()
        self.update_generate_state()

    def apply_mask_adjustments(self, mask, thickness, blur=0.0):
        binary = np.asarray(mask, dtype=np.uint8) >= 128
        adjusted_mask = adjust_mask_thickness(binary, thickness)
        adjusted = Image.fromarray((adjusted_mask * 255).astype(np.uint8), 'L')
        blur = float(blur)
        if blur < 0:
            raise ValueError('mask_blur cannot be negative.')
        if blur > 0:
            adjusted = adjusted.filter(ImageFilter.GaussianBlur(radius=blur))
        return adjusted

    def ensure_segmentation(self):
        if not hasattr(self, 'segmentation') or self.segmentation is None:
            self.segmentation = SegDinoSAM2()

    def load_segmentation(self):
        self.ensure_segmentation()
        self.segmentation.load_dino()
        self.segmentation.load_sam2()

    def unload_segmentation(self):
        if not hasattr(self, 'segmentation') or self.segmentation is None:
            return
        try:
            self.segmentation.unload_dino()
        except Exception:
            pass
        try:
            self.segmentation.unload_sam2()
        except Exception:
            pass
        gc.collect()
        if DEVICE == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def regenerate_mask(self, image):
        if image is None or self.processing or self.model_loading or self.classification_loading:
            return
        self.segmentation_loading = True
        self.reload_mask_btn.configure(state='disabled')
        self.generate_btn.configure(state='disabled')
        self.after(0, self.show_output)
        try:
            print('Building mask...')
            mask = self.make_mask(image)
            self.mask_image = mask.copy()
            self.mask_source = 'auto'
            self.after(0, self.show_input)
            self.after(0, self.show_output)
            print('Mask ready')
        except Exception as e:
            self.mask_image = None
            print(f'Mask error: {e}')
            self.after(0, lambda err=str(e): print(f'Mask Error: {err}'))
        finally:
            self.segmentation_loading = False
            self.after(0, lambda: self.reload_mask_btn.configure(state='normal' if self.original_image is not None and not self.processing else 'disabled'))
            self.after(0, self.update_crop_button_state)
            self.after(0, self.update_generate_state)

    def reload_mask(self):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading or self.manual_segment_loading:
            return
        if self.original_image is None:
            return
        if self.manual_segment_mode:
            self.disable_manual_segment_mode()
        try:
            self.sync_config()
        except Exception as e:
            print(f'Configuration Error: {e}')
            return
        self.mask_image = None
        self.mask_source = None
        self.sd_input_image = None
        self.output_image = None
        self.output_showing_original = False
        self.generate_btn.configure(state='disabled')
        self.reload_mask_btn.configure(state='disabled')
        self.update_manual_segment_buttons()
        self.show_input()
        self.show_output()
        print('Reloading mask...')
        threading.Thread(target=self.reload_mask_worker, daemon=True).start()

    def reload_mask_worker(self):
        try:
            processed = self.preprocess_uploaded_image(self.original_image)
            self.after(0, self.show_input)
            self.regenerate_mask(processed.copy())
        except Exception as e:
            print(f'RELOAD MASK error: {e}')
            self.after(0, lambda err=str(e): print(f'Mask Reload Error: {err}'))
            self.after(0, lambda: self.reload_mask_btn.configure(state='normal' if self.original_image is not None else 'disabled'))

    def preprocess_uploaded_image(self, image):
        base = image.convert('RGB').copy()
        self.input_image = base.copy()
        cropped = self.get_cropped_image(base)
        processed = self.prepare_sd_image(cropped)
        self.sd_input_image = processed.copy()
        print(f'Input ready {processed.width}x{processed.height}')
        return processed

    def make_mask(self, image):
        seg_pos_prompt = str(self.config.get('seg_pos_prompt', ''))
        seg_neg_prompt = str(self.config.get('seg_neg_prompt', ''))
        thickness = float(self.config.get('mask_thickness', 0.0))
        blur = float(self.config.get('mask_blur', 0.0))
        if thickness < 0:
            raise ValueError('mask_thickness cannot be negative.')
        if blur < 0:
            raise ValueError('mask_blur cannot be negative.')
        self.load_segmentation()
        try:
            print('Segmenting...')
            result = self.segmentation.segment(image, seg_pos_prompt, seg_neg_prompt, thickness=0)
            masks = getattr(result, 'masks', None)
            if masks is None and isinstance(result, dict):
                masks = result.get('masks')
            if masks is None:
                mask_image = getattr(result, 'mask_image', None)
                if mask_image is None and isinstance(result, dict):
                    mask_image = result.get('mask_image')
                if mask_image is None:
                    raise RuntimeError('segdinosam2 returned no masks.')
                raw_mask = Image.fromarray(np.asarray(mask_image, dtype=np.uint8), 'L')
                count = 1
            else:
                masks = [np.asarray(mask, dtype=bool) for mask in masks if np.asarray(mask).any()]
                if not masks:
                    raise RuntimeError('segdinosam2 returned no usable masks.')
                combined = np.zeros(image.size[::-1], dtype=bool)
                for mask in masks:
                    if mask.shape != combined.shape:
                        resized = Image.fromarray((mask.astype(np.uint8) * 255), 'L').resize(image.size, Image.Resampling.NEAREST)
                        mask = np.asarray(resized, dtype=np.uint8) > 0
                    combined |= mask
                raw_mask = Image.fromarray((combined.astype(np.uint8) * 255).astype(np.uint8), 'L')
                count = len(masks)
            if raw_mask.size != image.size:
                raw_mask = raw_mask.resize(image.size, Image.Resampling.NEAREST)
            mask = self.apply_mask_adjustments(raw_mask, thickness, blur)
            print(f'Mask ready • {count} mask(s)')
            return mask.copy()
        finally:
            self.unload_segmentation()
            self.cleanup_gpu()

    def has_valid_mask(self, mask):
        if mask is None:
            return False
        try:
            return np.asarray(mask, dtype=np.uint8).max() >= 10
        except Exception:
            return False

    def compose_mask_for_display(self, image, fitted_size):
        if self.mask_image is None:
            return None
        display_mask = Image.new('L', image.size, 0)
        crop_box = self.get_effective_crop_box()
        width, height = image.size
        x1 = max(0, min(width - 1, int(round(crop_box[0] * width))))
        y1 = max(0, min(height - 1, int(round(crop_box[1] * height))))
        x2 = max(x1 + 1, min(width, int(round(crop_box[2] * width))))
        y2 = max(y1 + 1, min(height, int(round(crop_box[3] * height))))
        crop_size = (x2 - x1, y2 - y1)
        mask = self.mask_image.resize(crop_size, Image.Resampling.NEAREST)
        display_mask.paste(mask, (x1, y1))
        return display_mask.resize(fitted_size, Image.Resampling.NEAREST)
