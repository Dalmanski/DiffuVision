import threading
from pathlib import Path
import numpy as np, imageio
from PIL import Image
from utils.dupfilename import prompt_save_path

SAVE_MODES = ['Output Image', 'Compare Image', 'Fade Vid', 'Tearing Vid', 'Eye Vid']
VID_TAGS = {'Fade Vid': 'FD', 'Tearing Vid': 'TR', 'Eye Vid': 'EY'}

class SaveOutputMixin:
    def save(self):
        if self.output_image is None:
            return
        mode = self.save_mode.get()
        if mode != SAVE_MODES[0]:
            return self.save_compare() if mode == SAVE_MODES[1] else self.save_vid()
        path = prompt_save_path('Save generated image', f'{Path(self.img_name or "image").stem}_DiffuVision', self.input_path, self.last_dir)
        if path is None:
            return
        self.last_dir = path.parent
        try:
            output = self.output_image
            if path.suffix.lower() in ('.jpg', '.jpeg'):
                output = output.convert('RGB')
            output.save(path)
            print('Saved')
        except Exception as e:
            print(f'Save Error: {e}')

    def save_compare(self):
        if self.original_image is None or self.output_image is None:
            return
        comparison_image = self.original_image if self.full_out.get() else (self.sd_input_image or self.original_image)
        path = prompt_save_path('Save comparison image', f'{Path(self.img_name or "image").stem}_DiffuVision_CF', self.input_path, self.last_dir)
        if path is None:
            return
        self.last_dir = path.parent
        try:
            original = comparison_image.convert('RGB')
            output = self.output_image.convert('RGB')
            target_height = max(original.height, output.height)
            left_width = int(original.width * target_height / original.height)
            right_width = int(output.width * target_height / output.height)
            original = original.resize((left_width, target_height), Image.Resampling.LANCZOS)
            output = output.resize((right_width, target_height), Image.Resampling.LANCZOS)
            comparison = Image.new('RGB', (left_width + right_width, target_height), (0, 0, 0))
            comparison.paste(original, (0, 0))
            comparison.paste(output, (left_width, 0))
            comparison.save(path)
            print('Comparison Saved')
        except Exception as e:
            print(f'Comparison Save Error: {e}')

    def save_vid(self):
        if self.original_image is None:
            return
        name = self.save_mode.get()
        path = prompt_save_path(f'Save {name.lower()}', f'{Path(self.img_name or "image").stem}_DiffuVision_{VID_TAGS[name]}', self.input_path, self.last_dir, ext='.mp4', filetypes=[('MP4', '*.mp4')])
        if path is None:
            return
        self.last_dir = path.parent
        out = self.output_image.convert('RGB')
        size = (out.width // 2 * 2, out.height // 2 * 2)
        orig = (self.original_image if self.full_out.get() else (self.sd_input_image or self.original_image)).convert('RGB')
        print(f'Rendering {name}...')
        trans = {'Fade Vid': lambda: (0.5, Image.blend, Image.blend), 'Tearing Vid': lambda: (0.8, self.claw_trans(size), self.claw_trans(size)), 'Eye Vid': lambda: (1.8, *[self.blink_trans(size)] * 2)}[name]()
        args = (path, name, orig.resize(size, Image.Resampling.LANCZOS), out.resize(size, Image.Resampling.LANCZOS), trans)
        threading.Thread(target=self.write_vid, args=args, daemon=True).start()

    @staticmethod
    def blink_trans(size, first=0.4, second=0.2):
        w, h = size
        y, x = np.ogrid[0:h, 0:w]
        dy = np.abs(y - h / 2).astype(np.float32)
        lid = (0.9 * h * (1 - 0.35 * ((x - w / 2) / (w / 2)) ** 2)).astype(np.float32)
        ease = lambda t: t * t * (3 - 2 * t)
        edges = np.cumsum([0, first / 2, first / 2, second / 2, second / 2]) / (first + second)
        def blink(a, b, t):
            k = min(int(np.searchsorted(edges, t, 'right')) - 1, 3)
            s = ease(min((t - edges[k]) / (edges[k + 1] - edges[k]), 1))
            alpha = np.clip(((s if k % 2 else 1 - s) * lid - dy) / (0.04 * h), 0, 1)[..., None]
            return Image.fromarray((np.asarray(b if k == 3 else a) * alpha).astype(np.uint8))
        return blink

    @staticmethod
    def claw_trans(size, swipes=3, marks=4):
        w, h = size
        rng = np.random.default_rng()
        diag = float(np.hypot(w, h))
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        grid = np.arange(int(2 * diag) + 1)
        noise = lambda n, amp: np.interp(grid, np.linspace(0, 2 * diag, n), rng.uniform(-amp, amp, n)).astype(np.float32)
        groups, kicks = [], []
        for g in range(swipes):
            ang = rng.uniform(0, np.pi)
            kicks.append((0.15 * g, ang + rng.choice([0, np.pi]), rng.choice([-1, 1])))
            cx, cy = w / 2 + rng.uniform(-0.08, 0.08) * diag, h / 2 + rng.uniform(-0.08, 0.08) * diag
            u, v = (x - cx) * np.cos(ang) + (y - cy) * np.sin(ang), (y - cy) * np.cos(ang) - (x - cx) * np.sin(ang)
            v -= rng.uniform(-0.3, 0.3) / diag * u * u
            groups.append((u, v, np.clip(u + diag, 0, 2 * diag).astype(np.int32), [((k - (marks - 1) / 2) * 0.045 * diag, rng.uniform(-0.04, 0.04) * diag, rng.uniform(0.3, 0.55) * diag,
                           noise(30, 0.35) + noise(140, 0.15), noise(30, 0.35) + noise(140, 0.15), noise(12, 0.006 * diag), 0.15 * g + 0.02 * k) for k in range(marks)]))
        ease = lambda t: t * t * (3 - 2 * t)
        def shake(img, t):
            amp, dx, dy, rot = 0, 0, 0, 0
            for t0, ang, spin in kicks:
                if t >= t0:
                    k = (t - t0) / 0.05
                    e = k * np.exp(1 - k)
                    amp, dx, dy, rot = amp + e, dx + e * np.cos(ang), dy + e * np.sin(ang), rot + e * spin
            if amp < 1e-3:
                return img
            tx, ty, sr = dx * 0.005 * diag, dy * 0.005 * diag, abs(np.sin(np.radians(0.3 * rot)))
            s = 1.005 + 2 * max((abs(tx) + h * sr / 2) / w, (abs(ty) + w * sr / 2) / h)
            img = img.rotate(0.3 * rot, Image.Resampling.BICUBIC, translate=(tx, ty))
            return img.resize((w, h), Image.Resampling.BICUBIC, box=(w * (1 - 1 / s) / 2, h * (1 - 1 / s) / 2, w * (1 + 1 / s) / 2, h * (1 + 1 / s) / 2))
        def claw(a, b, t):
            if t >= 1:
                return b
            hole = np.zeros((h, w), bool)
            grow = 0.6 * diag * ease(min(max((t - 0.4) / 0.6, 0), 1)) ** 3
            for u, v, ui, mks in groups:
                for off, shift, length, n1, n2, wob, delay in mks:
                    p = ease(min(max((t - delay) / 0.15, 0), 1))
                    if p <= 0:
                        continue
                    taper = np.sqrt(np.clip(1 - np.abs(u - shift) / (p * length + 2 * grow), 0, 1))
                    d = v - off - wob[ui]
                    hole |= np.abs(d) < (0.006 * diag + grow) * taper * (1 + np.where(d < 0, n1[ui], n2[ui]))
            return shake(Image.fromarray(np.where(hole[..., None], np.asarray(b), np.asarray(a))), t)
        return claw

    def write_vid(self, path, name, orig, out, trans, fps=30):
        w, h = out.size
        smooth = lambda t: t * t * (3 - 2 * t)
        zoom = lambda a, b, t, ease=smooth: out.resize((w, h), Image.Resampling.BICUBIC, box=tuple(x + (y - x) * ease(t) for x, y in zip(a, b)))
        full, top, bottom = (0, 0, w, h), (w / 4, 0, w * 3 / 4, h / 2), (w / 4, h / 2, w * 3 / 4, h)
        dur, trans_in, trans_out = trans
        steps = [(1, lambda t: orig), (dur, lambda t: trans_in(orig, out, t)), (0.5, lambda t: out),
                 (0.5, lambda t: zoom(full, top, t, lambda t: 1 - (1 - t) ** 3)), (3, lambda t: zoom(top, bottom, t)), (0.5, lambda t: zoom(bottom, full, t)),
                 (dur, lambda t: trans_out(out, orig, t)), (0.5, lambda t: orig)]
        try:
            with imageio.get_writer(path, fps=fps, codec='libx264', quality=8, macro_block_size=1) as writer:
                for dur, frame in steps:
                    n = round(dur * fps)
                    for i in range(n):
                        writer.append_data(np.asarray(frame((i + 1) / n)))
            print(f'{name} Saved')
        except Exception as e:
            print(f'{name} Save Error: {e}')
