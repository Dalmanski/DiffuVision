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