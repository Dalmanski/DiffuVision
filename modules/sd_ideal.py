from PIL import Image, ImageOps

MAX_SIDE = 768
OUTPUT_TARGET = 1080

RECOMMENDED_RATIO_SIZES = {
    '1:1': (512, 512),
    '4:3': (592, 440),
    '3:2': (624, 416),
    '16:9': (680, 384),
    '5:4': (568, 456),
    '4:5': (456, 568),
    '3:4': (440, 592),
    '2:3': (416, 624),
    '9:16': (384, 680),
}

class SDIdealImageMixin:
    def resize_image(self, image):
        image = image.convert('RGB')
        width, height = image.size
        ratio = self.ratio_var.get().strip().upper()
        if ratio in RECOMMENDED_RATIO_SIZES:
            new_width, new_height = RECOMMENDED_RATIO_SIZES[ratio]
        else:
            longest = max(width, height)
            scale = min(1.0, MAX_SIDE / longest)
            new_width = max(8, int(round(width * scale / 8)) * 8)
            new_height = max(8, int(round(height * scale / 8)) * 8)
        if (new_width, new_height) == (width, height):
            return image
        print(f'Resize → {new_width}x{new_height}')
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def prepare_sd_image(self, image):
        if not self.recommended_sd_var.get():
            return ImageOps.exif_transpose(image).convert('RGB')
        return self.resize_image(self.filter_sd_inpainting_image(image))

    def filter_sd_inpainting_image(self, image):
        image = ImageOps.exif_transpose(image)
        if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert('RGB')
        else:
            image = image.convert('RGB')
        width, height = image.size
        max_size = 768
        scale = min(1.0, max_size / max(width, height))
        width = max(64, int(round(width * scale / 8)) * 8)
        height = max(64, int(round(height * scale / 8)) * 8)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image

    def resize_output(self, image):
        width, height = image.size
        longest = max(width, height)
        if longest <= OUTPUT_TARGET:
            return image
        scale = OUTPUT_TARGET / longest
        new_size = (max(8, int(round(width * scale))), max(8, int(round(height * scale))))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def overlay_generated_crop(self, original_image, generated_crop, crop_box):
        original = original_image.convert('RGB').copy()
        width, height = original.size
        left, top, right, bottom = crop_box
        x1 = max(0, min(width - 1, int(round(left * width))))
        y1 = max(0, min(height - 1, int(round(top * height))))
        x2 = max(x1 + 1, min(width, int(round(right * width))))
        y2 = max(y1 + 1, min(height, int(round(bottom * height))))
        generated_crop = generated_crop.resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
        original.paste(generated_crop, (x1, y1))
        return original

    def generation_size(self, width, height):
        width = max(8, int(width))
        height = max(8, int(height))
        ratio = self.ratio_var.get().strip().upper()
        if self.recommended_sd_var.get() and ratio in RECOMMENDED_RATIO_SIZES:
            new_width, new_height = RECOMMENDED_RATIO_SIZES[ratio]
        elif self.recommended_sd_var.get():
            longest = max(width, height)
            scale = min(1.0, MAX_SIDE / longest)
            new_width = max(8, int(round(width * scale / 8)) * 8)
            new_height = max(8, int(round(height * scale / 8)) * 8)
        else:
            new_width = max(64, (width // 8) * 8)
            new_height = max(64, (height // 8) * 8)
        print(f'Size {new_width}x{new_height} • {ratio}')
        return new_width, new_height