from pathlib import Path

from PIL import Image


def load_image(path):
    try:
        with Image.open(path) as image:
            return image.copy()
    except (OSError, Image.UnidentifiedImageError) as pillow_error:
        try:
            import av
        except ImportError:
            raise pillow_error

        container = av.open(str(Path(path)))
        try:
            frame = next(container.decode(video=0))
            return frame.to_image()
        finally:
            container.close()


def flatten_alpha(image):
    if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert('RGB'), True
    return image.convert('RGB'), False


def pixel_box(box, size):
    width, height = size
    left, top, right, bottom = box
    x1 = max(0, min(width - 1, int(round(left * width))))
    y1 = max(0, min(height - 1, int(round(top * height))))
    x2 = max(x1 + 1, min(width, int(round(right * width))))
    y2 = max(y1 + 1, min(height, int(round(bottom * height))))
    return x1, y1, x2, y2
