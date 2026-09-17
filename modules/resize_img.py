from PIL import Image
from modules.segment_img import MAX_SIDE, OUTPUT_TARGET

RECOMMENDED_RATIO_SIZES = {'1:1': (512, 512), '4:3': (768, 576), '3:2': (768, 512), '16:9': (768, 432), '5:4': (640, 512), '4:5': (512, 640), '3:4': (576, 768), '2:3': (512, 768), '9:16': (432, 768)}

class ResizeImageMixin:
    def resize_image(self, image):
        image = image.convert('RGB')
        w, h = image.size
        ratio = self.ratio_var.get().strip().upper()
        if ratio in RECOMMENDED_RATIO_SIZES:
            nw, nh = RECOMMENDED_RATIO_SIZES[ratio]
        else:
            longest = max(w, h)
            if longest <= MAX_SIDE:
                nw = max(8, int(round(w / 8) * 8))
                nh = max(8, int(round(h / 8) * 8))
            else:
                scale = MAX_SIDE / longest
                nw = max(8, int(round(w * scale / 8) * 8))
                nh = max(8, int(round(h * scale / 8) * 8))
        if (nw, nh) == (w, h):
            return image
        self.console.log(f'Resize → {nw}x{nh}')
        return image.resize((nw, nh), Image.Resampling.LANCZOS)

    def resize_output(self, image):
        w, h = image.size
        longest = max(w, h)
        if longest <= OUTPUT_TARGET:
            return image
        scale = OUTPUT_TARGET / longest
        nw = max(8, int(round(w * scale)))
        nh = max(8, int(round(h * scale)))
        return image.resize((nw, nh), Image.Resampling.LANCZOS)

    def overlay_generated_crop(self, original_image, generated_crop, crop_box):
        original = original_image.convert('RGB').copy()
        width, height = original.size
        left, top, right, bottom = crop_box
        x1 = max(0, min(width - 1, int(round(left * width))))
        y1 = max(0, min(height - 1, int(round(top * height))))
        x2 = max(x1 + 1, min(width, int(round(right * width))))
        y2 = max(y1 + 1, min(height, int(round(bottom * height))))
        target_size = (x2 - x1, y2 - y1)
        generated_crop = generated_crop.resize(target_size, Image.Resampling.LANCZOS)
        original.paste(generated_crop, (x1, y1))
        return original

    def generation_size(self, width, height):
        width = max(8, int(width))
        height = max(8, int(height))
        ratio = self.ratio_var.get().strip().upper()
        if self.resize_var.get() and ratio in RECOMMENDED_RATIO_SIZES:
            w, h = RECOMMENDED_RATIO_SIZES[ratio]
        elif self.resize_var.get():
            longest = max(width, height)
            if longest <= MAX_SIDE:
                w = max(8, int(round(width / 8) * 8))
                h = max(8, int(round(height / 8) * 8))
            else:
                scale = MAX_SIDE / longest
                w = max(8, int(round(width * scale / 8) * 8))
                h = max(8, int(round(height * scale / 8) * 8))
        else:
            w = max(64, (width // 8) * 8)
            h = max(64, (height // 8) * 8)
        self.console.log(f'Size {w}x{h} • {ratio}')
        return w, h
