import threading, urllib.request, contextlib, io, tempfile, shutil
from pathlib import Path
import numpy as np, imageio, cv2
from PIL import Image
from utils.dupfilename import prompt_save_path
from utils.model_utils import clear_gpu
import modules.img_classify as img_classify

SAVE_MODES = ['Output Image', 'Compare Image', 'Fade Vid', 'Tearing Vid', 'Eye Vid', 'Crop Vid', 'Down Vid', 'Alive Vid']
ANIME_CASCADE_URL = 'https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml'
ANIME_CASCADE = Path(__file__).resolve().parent / 'lbpcascade_animeface.xml'
VID_TAGS = {'Fade Vid': 'FD', 'Tearing Vid': 'TR', 'Eye Vid': 'EY', 'Crop Vid': 'CR', 'Down Vid': 'DN'}

_DEPTHFLOW_SCENE = None

def depthflow_estimator():
    from depthflow.estimators.anything import DepthAnythingV2
    return DepthAnythingV2(model='base')

def depthflow_scene():
    global _DEPTHFLOW_SCENE
    if _DEPTHFLOW_SCENE is None:
        import math, os, random, shutil, dearlog, imageio_ffmpeg, depthflow
        from attrs import Factory, define
        from depthflow.scene import DepthScene
        dearlog.logger.handlers.clear()
        if shutil.which('ffmpeg') is None:
            bin_dir = depthflow.dirs.user_cache_path / 'bin'
            bin_dir.mkdir(parents=True, exist_ok=True)
            exe = bin_dir / 'ffmpeg.exe'
            if not exe.exists():
                try:
                    os.link(imageio_ffmpeg.get_ffmpeg_exe(), exe)
                except OSError:
                    shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), exe)
            os.environ['PATH'] = str(bin_dir) + os.pathsep + os.environ['PATH']
        @define
        class AliveScene(DepthScene):
            subject: float = 0.5
            phases: list = Factory(lambda: [random.uniform(0, 2 * math.pi) for _ in range(12)])
            video: object = None
            current: list = Factory(lambda: [None, None, 0])
            def follow_video(self):
                frame, depth, index = self.current
                while index < self.frame:
                    upcoming = next(self.video, None)
                    if upcoming is None:
                        break
                    index += 1
                    if not np.array_equal(upcoming[0], frame):
                        frame, depth = upcoming[0], self.estimator._post(upcoming[1][..., 0].astype(np.float32) / 255)
                        h, w = depth.shape[:2]
                        self.subject = 0.85 * self.subject + 0.15 * float(np.median(depth[h // 4:h * 3 // 4, w // 4:w * 3 // 4]))
                        self.image.from_numpy(frame)
                        self.depth.from_numpy(depth)
                self.current[:] = [frame, depth, index]
            def wave(self, first, *harmonics):
                return sum(a * math.sin(k * self.cycle + self.phases[(first + i) % len(self.phases)]) for i, (k, a) in enumerate(harmonics))
            def update(self):
                if self.video is not None:
                    self.follow_video()
                push = (1 - math.cos(self.cycle)) / 2
                sway = self.wave(0, (1, 0.6), (2, 0.3), (3, 0.1))
                shake = self.wave(3, (9, 0.5), (13, 0.3), (17, 0.2))
                self.state.height = 0.2
                self.state.isometric = 0.65
                self.state.steady = self.state.focus = self.subject
                self.state.dolly = 1.0 * push
                self.state.zoom = 0.93
                self.state.offset = (0.3 * sway + 0.03 * shake, 0.14 * self.wave(6, (1, 0.6), (2, 0.4)) + 0.04 * math.sin(10 * self.cycle) + 0.02 * self.wave(9, (11, 0.6), (15, 0.4)))
        _DEPTHFLOW_SCENE = AliveScene
    return _DEPTHFLOW_SCENE

class SaveOutputMixin:
    def save(self):
        if self.output_image is None:
            return
        mode = self.save_mode.get()
        if mode != SAVE_MODES[0]:
            return self.save_compare() if mode == SAVE_MODES[1] else self.save_alive() if mode == 'Alive Vid' else self.save_vid()
        path = prompt_save_path('Save generated image', f'{Path(self.img_name or "image").stem}_DiffuVision', self.input_path, self.last_dir)
        if path is None:
            return
        self.last_dir = path.parent
        try:
            output = self.output_image
            if path.suffix.lower() in ('.jpg', '.jpeg'):
                output = output.convert('RGB')
            output.save(path)
            print('[SAVE] Saved')
        except Exception as e:
            print(f'[SAVE ERROR] {e}')

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
            print('[SAVE] Comparison Saved')
        except Exception as e:
            print(f'[SAVE ERROR] {e}')

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
        print(f'[RENDER] Rendering {name}...')
        trans = {'Fade Vid': lambda parts: (0.5, Image.blend, Image.blend), 'Tearing Vid': lambda parts: (0.8, self.claw_trans(size), self.claw_trans(size)), 'Eye Vid': lambda parts: (1.8, *[self.blink_trans(size)] * 2),
                 'Crop Vid': lambda parts: (1.5, *[self.crop_trans(size, (self.body_point(parts, 0.75) or (0, size[1] / 2))[1])] * 2),
                 'Down Vid': None}[name]
        args = (path, name, orig.resize(size, Image.Resampling.LANCZOS), out.resize(size, Image.Resampling.LANCZOS), trans)
        threading.Thread(target=self.write_vid, args=args, daemon=True).start()

    def save_alive(self):
        path = prompt_save_path('Save alive vid', f'{Path(self.img_name or "image").stem}_DiffuVision_AV', self.input_path, self.last_dir, ext='.mp4', filetypes=[('MP4', '*.mp4')])
        if path is None:
            return
        self.last_dir = path.parent
        out = self.output_image.convert('RGB')
        out = out.crop((0, 0, out.width // 2 * 2, out.height // 2 * 2))
        print('[RENDER] Rendering Alive Vid...')
        threading.Thread(target=self.write_alive, args=(path, out), daemon=True).start()

    @classmethod
    def write_alive(cls, path, img):
        try:
            cls.write_depthflow(path, img)
            print('[SAVE] Alive Vid Saved')
        except Exception as e:
            print(f'[SAVE ERROR] {e}')

    @staticmethod
    def depth_maps(*imgs):
        estimator = None
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                from transformers.utils import logging as hf_logging
                depthflow_scene()
                estimator, verbosity = depthflow_estimator(), hf_logging.get_verbosity()
                hf_logging.set_verbosity_error()
                try:
                    estimator.load_model()
                    maps = [estimator.normalize(estimator._estimate(np.asarray(img)), dtype=np.uint8, min=0, max=255) for img in imgs]
                finally:
                    hf_logging.set_verbosity(verbosity)
            return [Image.fromarray(depth).resize(img.size, Image.Resampling.BICUBIC).convert('RGB') for depth, img in zip(maps, imgs)]
        finally:
            estimator = None
            clear_gpu()

    @staticmethod
    def alive_video(source, depth_source, path, seconds, fps=30):
        scene, readers = None, [imageio.get_reader(source), imageio.get_reader(depth_source)]
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                scene = depthflow_scene()(backend='headless', estimator=depthflow_estimator())
                frames = zip(*readers)
                first, first_depth = next(frames)
                depth = scene.estimator._post(first_depth[..., 0].astype(np.float32) / 255)
                h, w = depth.shape[:2]
                scene.subject = float(np.median(depth[h // 4:h * 3 // 4, w // 4:w * 3 // 4]))
                scene.input(image=first, depth=depth)
                scene.video, scene.current = frames, [first, depth, 0]
                scene.main(output=str(path), width=None, height=None, fps=fps, time=seconds, ssaa=1.5, quality=80)
        finally:
            for reader in readers:
                reader.close()
            scene = None
            clear_gpu()

    @classmethod
    def write_depthflow(cls, path, img, seconds=5, fps=30):
        scene = None
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                from transformers.utils import logging as hf_logging
                scene, verbosity = depthflow_scene()(backend='headless', estimator=depthflow_estimator()), hf_logging.get_verbosity()
                hf_logging.set_verbosity_error()
                try:
                    depth = scene.estimator.estimate(np.asarray(img))
                finally:
                    hf_logging.set_verbosity(verbosity)
                parts = cls.parse_body(img)
                h, w = depth.shape[:2]
                body = cv2.resize(np.logical_or.reduce(list(parts.values())).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) if parts else None
                scene.subject = float(np.median(depth[body] if body is not None and body.any() else depth[h // 4:h * 3 // 4, w // 4:w * 3 // 4]))
                scene.input(image=img, depth=depth)
                scene.main(output=str(path), width=None, height=None, fps=fps, time=seconds, ssaa=1.5, quality=80)
        finally:
            scene = None
            clear_gpu()

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
    def reveal_trans(size):
        w, h = size
        rng = np.random.default_rng()
        xs = np.arange(w)
        noise = lambda n, amp: np.interp(xs, np.linspace(0, w - 1, n), rng.uniform(-amp, amp, n)).astype(np.float32)
        edge, lead = noise(6, 0.08 * h) + noise(25, 0.03 * h) + noise(120, 0.008 * h), noise(5, 0.06 * h)
        y = np.arange(h, dtype=np.float32)[:, None]
        feather = max(h / 400, 1)
        def reveal(a, b, front, wobble=0):
            alpha = np.clip((front + edge + lead * wobble - y) / feather + 0.5, 0, 1)[..., None]
            return Image.fromarray((np.asarray(b) * alpha + np.asarray(a) * (1 - alpha)).astype(np.uint8))
        return reveal

    @staticmethod
    def crop_trans(size, cy):
        w, h = size
        y = np.arange(h)[:, None, None]
        core, edge = max(h / 300, 2) / 2, max(h / 600, 1)
        def crop(a, b, t):
            s = 1 - t
            top, bottom = cy * (1 - s), cy + (h - cy) * s
            frame = np.where((y >= top) & (y < bottom), np.asarray(a), np.asarray(b))
            if 0 < t < 1:
                dist = np.minimum(np.abs(y + 0.5 - top), np.abs(y + 0.5 - bottom))
                frame = np.where(dist <= core + edge, 0, frame)
                frame = np.where(dist <= core, 255, frame).astype(np.uint8)
            return Image.fromarray(frame)
        return crop

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
            groups.append((u, v, np.clip(u + diag, 0, 2 * diag).astype(np.int32), [((k - (marks - 1) / 2) * 0.045 * diag, rng.uniform(-0.04, 0.04) * diag, rng.uniform(0.3, 0.55) * diag, noise(30, 0.35) + noise(140, 0.15), noise(30, 0.35) + noise(140, 0.15), noise(12, 0.006 * diag), 0.15 * g + 0.02 * k) for k in range(marks)]))
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

    @staticmethod
    def find_face(img):
        try:
            points = img_classify._landmarks(img)
        except Exception:
            points = None
        if points and (points[0].visibility or 0) > 0.5:
            head = np.array([[p.x * img.width, p.y * img.height] for p in points[:11]])
            (x0, y0), (x1, y1) = head.min(axis=0), head.max(axis=0)
            size = max(x1 - x0, y1 - y0) * 1.3
            if size > 16:
                return (x0 + x1 - size) / 2, (y0 + y1 - size) / 2, size, size
        if not ANIME_CASCADE.exists():
            try:
                urllib.request.urlretrieve(ANIME_CASCADE_URL, ANIME_CASCADE)
            except Exception as e:
                print(f'[CASCADE ERROR] {e}')
        gray = cv2.equalizeHist(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY))
        min_size = (img.width // 20, img.height // 20)
        for cascade in (str(ANIME_CASCADE), *(cv2.data.haarcascades + c for c in ('haarcascade_frontalface_default.xml', 'haarcascade_profileface.xml'))):
            faces = cv2.CascadeClassifier(cascade).detectMultiScale(gray, 1.1, 5, minSize=min_size)
            if len(faces):
                return max(faces, key=lambda f: f[2] * f[3])
        return None

    @staticmethod
    def parse_body(img):
        parser, parts = None, {}
        try:
            import torch.nn.functional as F
            from modules.segdinosam2 import FashnHumanParser
            parser = FashnHumanParser()
            with contextlib.redirect_stdout(io.StringIO()):
                parser.load()
            _, outputs = parser.run(lambda: parser.processor(images=img, return_tensors='pt'))
            pred = F.interpolate(outputs.logits, size=(img.height, img.width), mode='bilinear', align_corners=False).argmax(dim=1)[0].cpu().numpy()
            for label, ids in parser.class_ids.items():
                mask = np.isin(pred, ids)
                if label != 'background' and mask.sum() >= 0.002 * mask.size:
                    parts[label] = mask
        except Exception as e:
            print(f'[FASHN ERROR] {e}')
        finally:
            if parser is not None:
                parser.unload()
        return parts

    @staticmethod
    def body_point(parts, k):
        rows = lambda *labels: [np.nonzero(parts[l])[0] for l in labels if l in parts]
        span = lambda *labels: np.concatenate(rows(*labels)) if rows(*labels) else None
        face, upper, garment, dress, legs = span('face'), span('torso', 'top'), span('pants', 'skirt'), span('dress'), span('legs')
        chin = face.max() if face is not None else next((y.min() for y in (upper, dress) if y is not None), None)
        waist = garment.min() if garment is not None else upper.max() if upper is not None else dress.min() + 0.4 * np.ptp(dress) if dress is not None else legs.min() if legs is not None else None
        if chin is None or waist is None or waist <= chin:
            return None
        y = chin + k * (waist - chin)
        body = np.logical_or.reduce(list(parts.values()))
        band = body[max(int(y - 0.05 * body.shape[0]), 0):int(y + 0.05 * body.shape[0]) + 1]
        xs = np.nonzero(band)[1]
        return (float(xs.mean()) if len(xs) else body.shape[1] / 2), float(y)

    @staticmethod
    def body_sides(parts, size):
        w, h = size
        keys = [k for k in ('torso', 'top', 'dress', 'skirt', 'pants', 'belt', 'legs', 'feet') if k in parts]
        body = np.logical_or.reduce([parts[k] for k in keys]) if keys else np.zeros((h, w), bool)
        rows, idx = body.any(axis=1), np.arange(h)
        if not rows.any():
            return np.full(h, w * 0.3), np.full(h, w * 0.7)
        left = np.interp(idx, idx[rows], body.argmax(axis=1)[rows])
        right = np.interp(idx, idx[rows], (w - 1 - body[:, ::-1].argmax(axis=1))[rows])
        k = max(int(0.04 * h), 1)
        blur = lambda a: np.convolve(np.pad(a, k, mode='edge'), np.ones(2 * k + 1) / (2 * k + 1), mode='valid')
        return blur(left), blur(right)

    @classmethod
    def head_points(cls, img, parts):
        if 'face' in parts:
            ys, xs = np.nonzero(parts['face'])
            fx0, fx1, fy0, fy1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
        else:
            face = cls.find_face(img)
            if face is None:
                return None
            fx0, fy0, fw, fh = map(float, face)
            fx1, fy1 = fx0 + fw, fy0 + fh
        fw, fh = fx1 - fx0, fy1 - fy0
        crown = fy0 - 0.3 * fh
        hair = [parts[k][:, int(fx0):int(fx1) + 1] for k in ('hair', 'hat') if k in parts]
        if hair:
            rows = np.nonzero(np.logical_or.reduce(hair).any(axis=1))[0]
            rows = rows[rows < fy0]
            if len(rows):
                crown = float(rows.min())
        y = crown + 0.5 * (fy0 - crown) + 0.1 * fh
        return np.array([fx0 + 0.22 * fw, y]), np.array([fx1 - 0.22 * fw, y]), fw

    @staticmethod
    def draw_hands(img, hands, depth=False):
        frame = np.asarray(img, dtype=np.float32).copy()
        h, w = frame.shape[:2]
        for x, y, r, lift, *soft in hands:
            if not soft or not any(soft[0]):
                continue
            (press, drag_x, drag_y), reach = soft[0], 4 * r
            x0, x1, y0, y1 = int(max(x - reach, 0)), int(min(x + reach, w)), int(max(y - reach, 0)), int(min(y + reach, h))
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
            dx, dy = xx - x, yy - y
            d = np.hypot(dx, dy) + 1e-3
            pull = np.exp(-(d / (2.2 * r)) ** 2) * np.clip(1 - (d / reach) ** 2, 0, 1)
            dent = press * np.exp(-(d / (1.5 * r)) ** 2)
            sink = 0.06 * r * dent * np.clip(d / r, 0, 1)
            frame[y0:y1, x0:x1] = cv2.remap(frame, xx - 0.5 - drag_x * pull + dx / d * sink, yy - 0.5 - drag_y * pull + dy / d * sink, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] - (9 * dent)[..., None] if depth else frame[y0:y1, x0:x1] * (1 - 0.03 * dent)[..., None]
        boxes = []
        for x, y, r, lift, *_ in hands:
            x0, x1, y0, y1 = int(max(x - 2 * r, 0)), int(min(x + 2.5 * r, w)), int(max(y - 2 * r, 0)), int(min(y + 2.5 * r, h))
            boxes.append((x0, x1, y0, y1, *np.mgrid[y0:y1, x0:x1].astype(np.float32)[::-1] + 0.5) if x0 < x1 and y0 < y1 else None)
        if not depth:
            for (x, y, r, lift, *_), box in zip(hands, boxes):
                if box is None:
                    continue
                x0, x1, y0, y1, xx, yy = box
                sx, sy = x + (0.1 + 0.3 * lift) * r, y + (0.18 + 0.5 * lift) * r
                shadow = (0.45 - 0.25 * lift) * np.clip(1 - (np.hypot(xx - sx, yy - sy) - 0.85 * r) / (r * (0.3 + 0.5 * lift)), 0, 1) ** 2
                frame[y0:y1, x0:x1] *= 1 - shadow[..., None]
        for (x, y, r, lift, *_), box in zip(hands, boxes):
            if box is None:
                continue
            x0, x1, y0, y1, xx, yy = box
            d = np.hypot(xx - x, yy - y)
            alpha = np.clip(r - d + 0.5, 0, 1)[..., None]
            region = frame[y0:y1, x0:x1]
            if depth:
                inside = region[alpha[..., 0] > 0]
                if not len(inside):
                    continue
                color = inside.max()
            else:
                shade = 0.8 + 0.2 * np.clip(1 - np.hypot(xx - x + 0.35 * r, yy - y + 0.35 * r) / (1.3 * r), 0, 1)
                shade *= 1 - 0.18 * np.clip((d - 0.82 * r) / (0.18 * r), 0, 1)
                color = shade[..., None] * np.array([255, 248, 242], np.float32)
            frame[y0:y1, x0:x1] = region * (1 - alpha) + color * alpha
        return Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))

    def down_steps(self, out, parts):
        w, h = out.size
        reveal, margin = self.reveal_trans((w, h)), 0.2 * h
        smooth = lambda t: t * t * (3 - 2 * t)
        mj = lambda s: s ** 3 * (10 - 15 * s + 6 * s * s)
        left, right = self.body_sides(parts, (w, h))
        head = self.head_points(out, parts)
        r = float(np.clip(0.33 * head[2] if head else 0.05 * min(w, h), 0.025 * min(w, h), 0.08 * min(w, h)))
        uppers = [parts[k] for k in ('torso', 'top', 'dress') if k in parts]
        upper_y = float(np.nonzero(np.logical_or.reduce(uppers).any(axis=1))[0].min()) if uppers else h * 0.3
        point = lambda k, default: (self.body_point(parts, k) or (0, default))[1]
        stomach_y = point(0.75, h * 0.55)
        if 'feet' in parts:
            feet_y = float(np.nonzero(parts['feet'].any(axis=1))[0].mean())
        elif 'legs' in parts:
            feet_y = float(np.nonzero(parts['legs'].any(axis=1))[0].max()) - r
        else:
            feet_y = h
        shoulder_y, chest_y, stomach_y, hips_y, feet_y = np.maximum.accumulate([upper_y + 0.15 * (stomach_y - upper_y), point(0.35, h * 0.4), stomach_y, point(1.15, h * 0.68), min(feet_y, h - 1.3 * r)])
        if head is None:
            y = max(upper_y - 0.12 * h, r)
            head = (np.array([w / 2 - 0.07 * w, y]), np.array([w / 2 + 0.07 * w, y]))

        def side(y, k=0.18, gap=1.2):
            i = int(np.clip(round(y), 0, h - 1))
            c, half = (left[i] + right[i]) / 2, max((right[i] - left[i]) * (0.5 - k), gap * r)
            return np.array([c - half, y]), np.array([c + half, y])

        def travel(a, b):
            def hands(t):
                spots = []
                for j, sign in enumerate((-1, 1)):
                    s = float(np.clip(t * 1.08 - 0.08 * j, 0, 1))
                    d = b[j] - a[j]
                    n = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-6)
                    n = n if n[0] * sign >= 0 else -n
                    e, lift = mj(s), np.sin(np.pi * s)
                    p = a[j] + d * e + n * 0.12 * np.hypot(*d) * np.sin(np.pi * e)
                    spots.append((p[0], p[1], r * (1 + 0.1 * lift), lift))
                return spots
            return hands

        def pat(a, pats=3):
            def hands(t):
                env, spots = np.sin(np.pi * t), []
                for j, sign in enumerate((-1, 1)):
                    beat = pats * t + 0.5 * j
                    lift = env * np.sin(np.pi * beat) ** 2
                    ang = 3 * np.pi * t * sign
                    swirl = 0.3 * r * env * np.array([np.cos(ang), np.sin(ang)])
                    p = a[j] + swirl - [0, 0.8 * r * lift]
                    age = (beat % 1) * 1.5 / pats
                    press = env * ((1 - lift) ** 2 + 0.2 * np.exp(-6 * age) * np.sin(22 * age))
                    spots.append((p[0], p[1], r * (0.96 + 0.14 * lift), lift, (press, *(0.2 * (1 - lift) * swirl))))
                return spots
            return hands

        def rub(a, turns=2):
            def hands(t):
                env, spots = mj(min(1, 4 * t, 4 * (1 - t))), []
                for j, sign in enumerate((-1, 1)):
                    ang = 2 * np.pi * turns * t * sign
                    swirl = 0.5 * r * env * np.array([np.cos(ang) * sign, np.sin(ang)])
                    p, pulse = a[j] + swirl, 0.5 + 0.5 * np.sin(2 * ang)
                    spots.append((p[0], p[1], r * (0.97 + 0.04 * env * pulse), 0, (env * (0.75 + 0.15 * pulse), *(0.22 * swirl))))
                return spots
            return hands

        def sweep(f0, f1, lo, hi, k_lo=0.18, k_hi=0.18):
            front = lambda t: f0 + (f1 - f0) * smooth(t)
            def hands(t):
                y = float(np.clip(front(t), lo, hi))
                return [(*p, r, 0) for p in side(y, k_lo + (k_hi - k_lo) * (y - lo) / max(hi - lo, 1))]
            return lambda t, o, b: reveal(o, b, front(t), np.sin(np.pi * t)), hands

        orig, done, hold = (lambda t, o, b: o), (lambda t, o, b: b), (lambda y: lambda t, o, b: reveal(o, b, y))
        start = (np.array([-1.5 * r, h + 1.5 * r]), np.array([w + 1.5 * r, h + 1.5 * r]))
        mid, half = (head[0] + head[1]) / 2, max((head[1][0] - head[0][0]) / 2, 1.25 * r)
        head = (mid - [half, 0], mid + [half, 0])
        shoulders, chest, stomach, feet = side(shoulder_y), side(chest_y, 0.27, 1.65), side(stomach_y, 0.27), side(feet_y)
        hips_x = sum(side(hips_y)) / 2
        hips = (hips_x - [1.65 * r, 0], hips_x + [1.65 * r, 0])
        def cam(at, zoom_in, amount=0.3):
            c = (at[0] + at[1]) / 2
            def box(t):
                z = 1 + amount * (mj(t) if zoom_in else 1 - mj(t))
                bw, bh = w / z, h / z
                x0, y0 = min(max(c[0] - bw / 2, 0), w - bw), min(max(c[1] - bh / 2, 0), h - bh)
                return (x0, y0, x0 + bw, y0 + bh)
            return box

        wipe = lambda f0, f1: sweep(f0, f1, f0, f1)[0]
        rest = lambda at: lambda t: [(*p, r, 0) for p in at]
        return [(0.8, orig), (1.4, orig, travel(start, head)), (1.5, orig, pat(head), cam(head, True)), (0.9, wipe(-margin, shoulder_y), travel(head, shoulders), cam(head, False)), (0.5, hold(shoulder_y), rest(shoulders)),
                (2.0, *sweep(shoulder_y, stomach_y, shoulder_y, stomach_y, 0.18, 0.27)), (0.9, hold(stomach_y), travel(stomach, chest)), (2.0, hold(stomach_y), rub(chest), cam(chest, True)), (0.9, hold(stomach_y), travel(chest, stomach), cam(chest, False)),
                (2.0, *sweep(stomach_y, feet_y, stomach_y, feet_y, 0.27, 0.18)), (0.9, wipe(feet_y, h + margin), travel(feet, hips)), (2.0, done, rub(hips), cam(hips, True)), (0.9, wipe(h + margin, feet_y), travel(hips, feet), cam(hips, False)),
                (2.5, *sweep(feet_y, -margin, -margin - 2 * r, feet_y)), (0.5, orig)]

    @classmethod
    def face_box(cls, img):
        face = cls.find_face(img)
        if face is None:
            return None
        w, h = img.size
        fx, fy, fw, fh = face
        s = min(max(fh * 3 / h, fw * 3 / w, 0.25), 0.5)
        cx, cy = min(max(fx + fw / 2, w * s / 2), w - w * s / 2), min(max(fy + fh / 2, h * s / 2), h - h * s / 2)
        return (cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2)

    def write_vid(self, path, name, orig, out, trans, fps=30):
        w, h = out.size
        smooth = lambda t: t * t * (3 - 2 * t)
        zoom = lambda img, a, b, t, ease=smooth: img.resize((w, h), Image.Resampling.BICUBIC, box=tuple(x + (y - x) * ease(t) for x, y in zip(a, b)))
        full = (0, 0, w, h)
        top = self.face_box(out) or self.face_box(orig) or (w / 4, 0, w * 3 / 4, h / 2)
        parts = self.parse_body(out)
        if name == 'Down Vid':
            steps = self.down_steps(out, parts)
        else:
            bw, bh, hip = top[2] - top[0], top[3] - top[1], self.body_point(parts, 1.15)
            x0, y0 = top[0], h - bh
            if hip:
                x0 = min(max(hip[0] - bw / 2, 0), w - bw)
                y0 = min(max(hip[1] - bh / 2, top[1]), h - bh)
            bottom = (x0, y0, x0 + bw, y0 + bh)
            dur, trans_in, trans_out = trans(parts)
            steps = [(1, lambda t, o, b: o), (dur, lambda t, o, b: trans_in(o, b, t)), (0.5, lambda t, o, b: b), (0.5, lambda t, o, b: zoom(b, full, top, t, lambda t: 1 - (1 - t) ** 3)), (3, lambda t, o, b: zoom(b, top, bottom, t)), (0.5, lambda t, o, b: zoom(b, bottom, full, t)), (dur, lambda t, o, b: trans_out(b, o, t)), (0.5, lambda t, o, b: o)]
        self.render_vid(path, name, orig, out, steps, fps)

    def render_vid(self, path, name, orig, out, steps, fps=30):
        temp = Path(tempfile.mkdtemp())
        try:
            try:
                depths = self.depth_maps(orig, out)
            except Exception as e:
                print(f'[ALIVE] Depth estimation failed: {e}')
                depths = None
            total, sources = 0, [(temp / 'still.mp4', (orig, out))] + ([(temp / 'depth.mp4', depths)] if depths else [])
            writers = [imageio.get_writer(file, fps=fps, codec='libx264', quality=10 if k else 9, macro_block_size=1) for k, (file, _) in enumerate(sources)]
            try:
                for dur, frame, *extra in steps:
                    hands, cam = (extra + [None, None])[:2]
                    n = round(dur * fps)
                    total += n
                    for i in range(n):
                        t = (i + 1) / n
                        spots, box = hands and hands(t), cam and cam(t)
                        for k, (writer, (_, images)) in enumerate(zip(writers, sources)):
                            img = frame(t, *images)
                            if spots:
                                img = self.draw_hands(img, spots, depth=k > 0)
                            if box:
                                img = img.resize(img.size, Image.Resampling.BICUBIC, box=box)
                            writer.append_data(np.asarray(img))
            finally:
                for writer in writers:
                    writer.close()
            try:
                if not depths:
                    shutil.move(temp / 'still.mp4', path)
                else:
                    self.alive_video(temp / 'still.mp4', temp / 'depth.mp4', path, total / fps, fps)
            except Exception as e:
                print(f'[ALIVE] Alive motion unavailable, saving without it: {e}')
                shutil.move(temp / 'still.mp4', path)
            print(f'[SAVE] {name} Saved')
        except Exception as e:
            print(f'[SAVE ERROR] {name}: {e}')
        finally:
            shutil.rmtree(temp, ignore_errors=True)