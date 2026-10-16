from PIL import Image, ImageOps
from utils.image_loader import flatten_alpha, pixel_box

MAX_SIDE = 768
OUTPUT_TARGET = 1080
PIXEL_VOLUME = 512 * 512

RATIO_SIZES = {
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

def limit_longest_side(image, target=OUTPUT_TARGET):
    width, height = image.size
    longest = max(width, height)
    if longest <= target:
        return image
    scale = target / longest
    new_size = (max(8, int(round(width * scale))), max(8, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)

class SDIdealImageMixin:
    def ratio_size(self, ratio):
        width = max(8, int(round((PIXEL_VOLUME * ratio) ** 0.5 / 8)) * 8)
        height = max(8, int(round((PIXEL_VOLUME / ratio) ** 0.5 / 8)) * 8)
        return width, height

    def recommended_size(self, width, height, ratio):
        if ratio in RATIO_SIZES:
            return RATIO_SIZES[ratio]
        if ratio == 'FREE':
            return self.ratio_size(width / height)
        scale = min(1.0, MAX_SIDE / max(width, height))
        return max(8, int(round(width * scale / 8)) * 8), max(8, int(round(height * scale / 8)) * 8)

    def resize_image(self, image):
        image = image.convert('RGB')
        new_width, new_height = self.recommended_size(*image.size, self.ratio_var.get().strip().upper())
        if (new_width, new_height) == image.size:
            return image
        print(f'[RESIZE] {new_width}x{new_height}')
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def prepare_sd_image(self, image):
        if not self.sd_rec.get():
            return ImageOps.exif_transpose(image).convert('RGB')
        return self.resize_image(self.filter_sd_image(image))

    def filter_sd_image(self, image):
        image, _ = flatten_alpha(ImageOps.exif_transpose(image))
        width, height = image.size
        scale = min(1.0, MAX_SIDE / max(width, height))
        width = max(64, int(round(width * scale / 8)) * 8)
        height = max(64, int(round(height * scale / 8)) * 8)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image

    def resize_output(self, image):
        return limit_longest_side(image)

    def overlay_crop(self, original_image, generated_crop, crop_box, mask=None):
        original = original_image.convert('RGB').copy()
        x1, y1, x2, y2 = pixel_box(crop_box, original.size)
        size = (x2 - x1, y2 - y1)
        generated_crop = generated_crop.resize(size, Image.Resampling.LANCZOS)
        if mask is None:
            original.paste(generated_crop, (x1, y1))
        else:
            original.paste(generated_crop, (x1, y1), mask.convert('L').resize(size, Image.Resampling.BILINEAR))
        return original

    def generation_size(self, width, height):
        width = max(8, int(width))
        height = max(8, int(height))
        ratio = self.ratio_var.get().strip().upper()
        if self.sd_rec.get():
            new_width, new_height = self.recommended_size(width, height, ratio)
        else:
            new_width = max(64, (width // 8) * 8)
            new_height = max(64, (height // 8) * 8)
        print(f'[SIZE] {new_width}x{new_height} • {ratio}')
        return new_width, new_height
