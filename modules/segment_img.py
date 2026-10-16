from PIL import Image, ImageFilter
import threading
import numpy as np
from modules.segdinosam2 import SegDinoSAM2, thicken_mask
from modules.selectsegsam2 import SAM2Segmenter
from utils.image_loader import pixel_box

def merge_masks(base, mask):
    if base.size != mask.size:
        base = base.resize(mask.size, Image.Resampling.NEAREST)
    return Image.fromarray(np.maximum(np.asarray(base, dtype=np.uint8), np.asarray(mask, dtype=np.uint8)), mode='L')

class SegmentImageMixin:

    def update_manual_btns(self):
        if not hasattr(self, 'manual_btn'):
            return
        base_state = 'normal' if self.original_image is not None and not self.input_is_busy() else 'disabled'
        self.manual_btn.configure(state=base_state)
        self.clear_seg_btn.configure(state='normal' if self.original_image is not None else 'disabled')
        if self.manual_mode:
            self.manual_btn.configure(fg_color='#1f8f3a', hover_color='#176b2c')
        else:
            self.manual_btn.reset_colors()

    def toggle_manual_mode(self):
        if self.input_is_busy(include_manual=False) or self.original_image is None:
            return
        if self.manual_mode:
            self.disable_manual_mode()
            print('[SAM2] SAM2 point selection disabled')
            return
        try:
            self.sync_config()
        except Exception as e:
            print(f'[CONFIG ERROR] {e}')
            return
        self.manual_mode = True
        self.manual_loading = True
        self.manual_btn.configure(state='disabled')
        self.clear_seg_btn.configure(state='normal')
        self.update_crop_state()
        print('[SAM2] Loading SAM2 point selection...')
        threading.Thread(target=self.load_sam_worker, daemon=True).start()

    def finish_manual_load(self):
        self.manual_loading = False
        self.after(0, self.update_manual_btns)
        self.after(0, self.update_crop_state)
        self.after(0, self.show_input)
        self.after(0, self.update_gen_state)

    def load_sam_worker(self):
        try:
            if self.sd_input_image is None:
                self.prep_input(self.original_image)
            image = self.sd_input_image.copy() if self.sd_input_image is not None else self.input_image.copy()
            self.manual_segmenter = SAM2Segmenter()
            self.manual_segmenter.load_image(image)
            self.manual_image = self.input_image.copy() if self.input_image is not None else self.original_image.copy()
            self.manual_base_mask = self.mask_image.copy() if self.has_valid_mask(self.mask_image) else None
            print('[SAM2] SAM2 point selection ready • click the image to append segments')
        except Exception as e:
            self.manual_mode = False
            print(f'[SAM2 ERROR] {e}')
            self.after(0, lambda err=str(e): print(f'[SAM2 ERROR] {err}'))
        finally:
            self.finish_manual_load()

    def disable_manual_mode(self):
        self.manual_mode = False
        self.manual_loading = False
        segmenter = self.manual_segmenter
        self.manual_segmenter = None
        self.manual_image = None
        self.manual_base_mask = None
        if segmenter is not None:
            segmenter.close()
        self.cleanup_gpu()
        self.reload_btn.configure(state='normal' if self.original_image is not None and not self.processing and not self.seg_loading else 'disabled')
        self.update_crop_state()
        self.show_input()

    def add_manual_segment(self, event):
        point = self.manual_point(event)
        if point is None:
            return
        image_x, image_y = point
        self.manual_loading = True
        self.update_manual_btns()
        threading.Thread(target=self.manual_worker, args=(image_x, image_y), daemon=True).start()

    def manual_point(self, event):
        if not self.manual_mode or self.manual_loading or self.manual_segmenter is None:
            return None
        if self.input_box is None or self.manual_image is None:
            return None
        x, y, width, height = self.input_box
        if event.x < x or event.x >= x + width or event.y < y or event.y >= y + height:
            return None
        image_width, image_height = self.manual_image.size
        source_x = max(0, min(image_width - 1, int((event.x - x) / width * image_width)))
        source_y = max(0, min(image_height - 1, int((event.y - y) / height * image_height)))
        crop_x1, crop_y1, crop_x2, crop_y2 = pixel_box(self.crop_or_full(), self.manual_image.size)
        if not crop_x1 <= source_x < crop_x2 or not crop_y1 <= source_y < crop_y2:
            return None
        segment_width, segment_height = self.manual_segmenter.get_image_size()
        image_x = min(segment_width - 1, int((source_x - crop_x1) / (crop_x2 - crop_x1) * segment_width))
        image_y = min(segment_height - 1, int((source_y - crop_y1) / (crop_y2 - crop_y1) * segment_height))
        return image_x, image_y

    def del_manual_segment(self, event):
        if self.processing or self.manual_loading:
            return
        point = self.manual_point(event)
        if point is None or not self.manual_segmenter.remove_segment_at(*point):
            return
        mask = self.manual_segmenter.get_mask()
        manual_mask = self.adjust_mask(mask * 255, *self.mask_settings()) if np.any(mask) else None
        if self.has_valid_mask(self.manual_base_mask) and manual_mask is not None:
            self.mask_image = merge_masks(self.manual_base_mask, manual_mask)
        else:
            self.mask_image = self.manual_base_mask.copy() if self.has_valid_mask(self.manual_base_mask) else manual_mask
        self.mask_src = 'manual' if self.mask_image is not None else None
        self.output_image = None
        self.show_orig = False
        self.show_input()
        self.show_output()
        self.update_gen_state()

    def manual_worker(self, image_x, image_y):
        try:
            mask = self.manual_segmenter.select_point(image_x, image_y, positive=True)
            mask = self.adjust_mask(mask, *self.mask_settings())
            if self.has_valid_mask(self.mask_image):
                mask = merge_masks(self.mask_image, mask)
            self.mask_image = mask
            self.mask_src = 'manual'
            self.output_image = None
            self.show_orig = False
            self.after(0, self.show_input)
            self.after(0, self.show_output)
        except Exception as e:
            print(f'[SAM2 ERROR] {e}')
            self.after(0, lambda err=str(e): print(f'[SAM2 ERROR] {err}'))
        finally:
            self.finish_manual_load()

    def clear_segments(self):
        if self.processing or self.manual_loading:
            return
        if self.manual_segmenter is not None:
            self.manual_segmenter.clear_all_segments()
        self.reset_preview_state()
        self.generate_btn.configure(state='disabled')
        print('[SEGMENT] All manually appended segments removed')
        self.show_input()
        self.show_output()
        self.update_gen_state()

    def mask_settings(self, config=None):
        config = self.config if config is None else config
        return float(config.get('mask_thickness', 0.0)), float(config.get('mask_blur', 0.0))

    def adjust_mask(self, mask, thickness, blur=0.0):
        binary = np.asarray(mask, dtype=np.uint8) >= 128
        adjusted_mask = thicken_mask(binary, thickness)
        adjusted = Image.fromarray((adjusted_mask * 255).astype(np.uint8), 'L')
        blur = float(blur)
        if blur < 0:
            raise ValueError('mask_blur cannot be negative.')
        if blur > 0:
            adjusted = adjusted.filter(ImageFilter.GaussianBlur(radius=blur))
        return adjusted

    def load_segmentation(self):
        if self.segmentation is None:
            self.segmentation = SegDinoSAM2()
        self.segmentation.load_dino()
        self.segmentation.load_sam2()

    def unload_segmentation(self):
        if self.segmentation is None:
            return
        for unload in (self.segmentation.unload_dino, self.segmentation.unload_sam2):
            try:
                unload()
            except Exception:
                pass

    def regenerate_mask(self, image):
        if image is None or self.processing or self.model_loading or self.class_loading:
            return
        self.seg_loading = True
        self.reload_btn.configure(state='disabled')
        self.generate_btn.configure(state='disabled')
        self.after(0, self.show_output)
        try:
            print('[MASK] Building mask...')
            mask = self.make_mask(image)
            self.mask_image = mask.copy()
            self.mask_src = 'auto'
            self.after(0, self.show_input)
            self.after(0, self.show_output)
            print('[MASK] Mask ready')
        except Exception as e:
            self.mask_image = None
            print(f'[MASK ERROR] {e}')
            self.after(0, lambda err=str(e): print(f'[MASK ERROR] {err}'))
        finally:
            self.seg_loading = False
            self.after(0, lambda: self.reload_btn.configure(state='normal' if self.original_image is not None and not self.processing else 'disabled'))
            self.after(0, self.update_crop_state)
            self.after(0, self.update_gen_state)

    def reload_mask(self):
        if self.input_is_busy() or self.original_image is None:
            return
        if self.manual_mode:
            self.disable_manual_mode()
        try:
            self.sync_config()
        except Exception as e:
            print(f'[CONFIG ERROR] {e}')
            return
        self.reset_preview_state()
        self.generate_btn.configure(state='disabled')
        self.reload_btn.configure(state='disabled')
        self.update_manual_btns()
        self.show_input()
        self.show_output()
        print('[MASK] Reloading mask...')
        threading.Thread(target=self.reload_mask_worker, daemon=True).start()

    def reload_mask_worker(self):
        try:
            processed = self.prep_input(self.original_image)
            self.after(0, self.show_input)
            self.regenerate_mask(processed.copy())
        except Exception as e:
            print(f'[MASK ERROR] {e}')
            self.after(0, lambda err=str(e): print(f'[MASK ERROR] {err}'))
            self.after(0, lambda: self.reload_btn.configure(state='normal' if self.original_image is not None else 'disabled'))

    def prep_input(self, image):
        base = image.convert('RGB').copy()
        self.input_image = base.copy()
        cropped = self.get_cropped_image(base)
        processed = self.prepare_sd_image(cropped)
        self.sd_input_image = processed.copy()
        print(f'[INPUT] Input ready {processed.width}x{processed.height}')
        return processed

    def make_mask(self, image, config=None):
        config = self.config if config is None else config
        seg_pos_prompt = str(config.get('seg_pos_prompt', ''))
        seg_neg_prompt = str(config.get('seg_neg_prompt', ''))
        thickness, blur = self.mask_settings(config)
        if thickness < 0:
            raise ValueError('mask_thickness cannot be negative.')
        if blur < 0:
            raise ValueError('mask_blur cannot be negative.')
        self.load_segmentation()
        try:
            print('[SEGMENT] Segmenting...')
            result = self.segmentation.segment(image, seg_pos_prompt, seg_neg_prompt, thickness=0)
            masks = [np.asarray(mask, dtype=bool) for mask in result.masks if np.asarray(mask).any()]
            if not masks:
                raise RuntimeError('segdinosam2 returned no usable masks.')
            combined = np.zeros(image.size[::-1], dtype=bool)
            for mask in masks:
                if mask.shape != combined.shape:
                    resized = Image.fromarray((mask.astype(np.uint8) * 255), 'L').resize(image.size, Image.Resampling.NEAREST)
                    mask = np.asarray(resized, dtype=np.uint8) > 0
                combined |= mask
            raw_mask = Image.fromarray(combined.astype(np.uint8) * 255, 'L')
            mask = self.adjust_mask(raw_mask, thickness, blur)
            print(f'[MASK] Mask ready • {len(masks)} mask(s)')
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

    def view_mask(self, image, fitted_size):
        if self.mask_image is None:
            return None
        display_mask = Image.new('L', image.size, 0)
        x1, y1, x2, y2 = pixel_box(self.crop_or_full(), image.size)
        mask = self.mask_image.resize((x2 - x1, y2 - y1), Image.Resampling.NEAREST)
        display_mask.paste(mask, (x1, y1))
        return display_mask.resize(fitted_size, Image.Resampling.NEAREST)
