import os
from pathlib import Path

from PIL import Image, ImageSequence, ImageTk


class GifAnimationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gif_jobs = {}

    def _resolve_gif(self, gif_path):
        gif = str(gif_path)
        if not os.path.isabs(gif):
            gif = str((Path(__file__).resolve().parents[1] / gif).resolve())
        return gif

    def _gif_next(self, gif_key):
        state = self._gif_jobs.get(gif_key)
        if not state:
            return
        canvas = state['canvas']
        frames = state['frames']
        state['index'] = (state['index'] + 1) % len(frames)
        canvas.itemconfigure(state['tag'], image=frames[state['index']])
        delay = max(20, int(40 / max(state['speed'], 0.05)))
        state['after_id'] = self.after(delay, lambda: self._gif_next(gif_key))

    def play_gif(self, gif_path, canvas=None, x=None, y=None, anchor='center', speed=1.0):
        canvas = canvas or getattr(self, 'in_canvas', None)
        if canvas is None:
            return None
        gif_key = (self._resolve_gif(gif_path), str(canvas))
        self.stop_gif(gif_path, canvas)
        try:
            frames = [ImageTk.PhotoImage(frame.copy()) for frame in ImageSequence.Iterator(Image.open(gif_key[0]))]
        except Exception:
            return None
        if not frames:
            return None
        if x is None or y is None:
            x = canvas.winfo_width() / 2
            y = canvas.winfo_height() / 2
        tag = f'gif_{abs(hash(gif_key))}'
        state = {'canvas': canvas, 'frames': frames, 'tag': tag, 'index': 0, 'after_id': None, 'speed': float(speed or 1.0)}
        self._gif_jobs[gif_key] = state
        canvas.create_image(x, y, image=frames[0], tag=tag, anchor=anchor)
        self._gif_next(gif_key)
        return tag

    def stop_gif(self, gif_path, canvas=None):
        canvas = canvas or getattr(self, 'in_canvas', None)
        if canvas is None:
            return
        gif_key = (self._resolve_gif(gif_path), str(canvas))
        state = self._gif_jobs.pop(gif_key, None)
        if state is None:
            canvas.delete(f'gif_{abs(hash(gif_key))}')
            return
        if state['after_id'] is not None:
            self.after_cancel(state['after_id'])
        canvas.delete(state['tag'])
