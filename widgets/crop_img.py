from utils.image_loader import pixel_box

FULL_BOX = 0.0, 0.0, 1.0, 1.0
MIN_CROP = 0.02

class CropImageMixin:
    @staticmethod
    def parse_ratio(choice):
        left, right = choice.split(':', 1)
        return float(left) / float(right)

    def crop_to_ratio(self, choice):
        if self.original_image is None or choice.strip().upper() == 'FREE':
            return FULL_BOX
        ratio = self.parse_ratio(choice)
        image_width, image_height = self.original_image.size
        image_aspect = image_width / image_height
        if image_aspect > ratio:
            crop_height = 1.0
            crop_width = ratio / image_aspect
        else:
            crop_width = 1.0
            crop_height = image_aspect / ratio
        left = (1.0 - crop_width) / 2.0
        top = (1.0 - crop_height) / 2.0
        return left, top, left + crop_width, top + crop_height

    def crop_or_full(self):
        return FULL_BOX if self.crop_box is None else self.crop_box

    def get_cropped_image(self, image):
        image = image.convert('RGB')
        if self.crop_box is None:
            return image.copy()
        return image.crop(pixel_box(self.crop_box, image.size))

    def clamp_crop_box(self, box):
        left, top, right, bottom = [float(value) for value in box]
        left = max(0.0, min(1.0 - MIN_CROP, left))
        top = max(0.0, min(1.0 - MIN_CROP, top))
        right = max(MIN_CROP, min(1.0, right))
        bottom = max(MIN_CROP, min(1.0, bottom))
        if right - left < MIN_CROP:
            if left + MIN_CROP <= 1.0:
                right = left + MIN_CROP
            else:
                left = max(0.0, right - MIN_CROP)
        if bottom - top < MIN_CROP:
            if top + MIN_CROP <= 1.0:
                bottom = top + MIN_CROP
            else:
                top = max(0.0, bottom - MIN_CROP)
        return left, top, right, bottom

    def is_free_ratio(self):
        return self.ratio_var.get().strip().upper() == 'FREE'

    def get_crop_ratio(self):
        if not self.is_free_ratio():
            return self.parse_ratio(self.ratio_var.get().strip().upper())
        if self.original_image is None:
            return None
        left, top, right, bottom = self.crop_or_full()
        crop_width = max(1.0, (right - left) * self.original_image.width)
        crop_height = max(1.0, (bottom - top) * self.original_image.height)
        return crop_width / crop_height

    def normalized_ratio(self, ratio):
        return ratio * self.original_image.height / self.original_image.width

    def clamp_fixed_box(self, box, ratio):
        if self.original_image is None or ratio is None:
            return self.clamp_crop_box(box)
        left, top, right, bottom = [float(value) for value in box]
        normalized_ratio = self.normalized_ratio(ratio)
        width = max(MIN_CROP, right - left)
        height = max(MIN_CROP, bottom - top)
        if width / height > normalized_ratio:
            height = width / normalized_ratio
        else:
            width = height * normalized_ratio
        width = min(width, 1.0, normalized_ratio)
        height = min(height, 1.0, 1.0 / normalized_ratio)
        if width < MIN_CROP:
            width = MIN_CROP
            height = width / normalized_ratio
        if height < MIN_CROP:
            height = MIN_CROP
            width = height * normalized_ratio
        if width > 1.0:
            width = 1.0
            height = width / normalized_ratio
        if height > 1.0:
            height = 1.0
            width = height * normalized_ratio
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        left = cx - width / 2.0
        right = cx + width / 2.0
        top = cy - height / 2.0
        bottom = cy + height / 2.0
        if left < 0.0:
            right -= left
            left = 0.0
        if right > 1.0:
            left -= right - 1.0
            right = 1.0
        if top < 0.0:
            bottom -= top
            top = 0.0
        if bottom > 1.0:
            top -= bottom - 1.0
            bottom = 1.0
        return self.clamp_crop_box((left, top, right, bottom))

    def clamp_for_ratio(self, box):
        if self.is_free_ratio():
            return self.clamp_crop_box(box)
        return self.clamp_fixed_box(box, self.get_crop_ratio())

    def mark_crop_changed(self, message):
        self.reset_preview_state()
        self.refresh_crop_ui(allow_reload=True)
        print(f'[CROP] {message}')
        self.show_input()
        self.show_output()

    def ratio_changed(self, choice):
        if self.input_is_busy(include_manual=False) or self.original_image is None:
            return
        if choice.strip().upper() == 'FREE':
            self.crop_box = FULL_BOX
        else:
            self.crop_box = self.clamp_fixed_box(self.crop_or_full(), self.get_crop_ratio())
        self.mark_crop_changed(f'Ratio {choice}')

    def crop_handle_size(self):
        return 11

    def crop_display_rect(self, display):
        x, y, width, height = display
        left, top, right, bottom = self.crop_or_full()
        return x + left * width, y + top * height, x + right * width, y + bottom * height

    def crop_handle_points(self, display):
        x1, y1, x2, y2 = self.crop_display_rect(display)
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        return {'nw': (x1, y1), 'n': (xm, y1), 'ne': (x2, y1), 'w': (x1, ym), 'e': (x2, ym), 'sw': (x1, y2), 's': (xm, y2), 'se': (x2, y2)}

    def get_crop_handle(self, event):
        if self.input_box is None:
            return None
        size = self.crop_handle_size() + 5
        for handle, point in self.crop_handle_points(self.input_box).items():
            if abs(event.x - point[0]) <= size and abs(event.y - point[1]) <= size:
                return handle
        return None

    def move_btn_size(self):
        return 34, 28

    def move_btn_box(self, display):
        _, y1, x2, _ = self.crop_display_rect(display)
        button_width, button_height = self.move_btn_size()
        return x2 - button_width, y1, x2, y1 + button_height

    def on_move_btn(self, event):
        if self.input_box is None:
            return False
        x1, y1, x2, y2 = self.move_btn_box(self.input_box)
        return x1 <= event.x <= x2 and y1 <= event.y <= y2

    def start_crop(self, event):
        if self.manual_mode:
            self.add_manual_segment(event)
            return
        if self.original_image is None or self.input_is_busy(include_manual=False) or self.input_box is None:
            return
        if self.on_move_btn(event):
            self.active_crop_handle = 'move'
            self.crop_box_start = self.crop_or_full()
            self.crop_move_start = (event.x, event.y)
            self.cropping = True
            self.draw_crop_overlay()
            return
        handle = self.get_crop_handle(event)
        if handle is None:
            return
        self.active_crop_handle = handle
        self.crop_box_start = self.crop_or_full()
        self.cropping = True

    def drag_crop(self, event):
        if self.manual_mode or not self.cropping or self.input_box is None or self.crop_box_start is None:
            return
        x, y, width, height = self.input_box
        left, top, right, bottom = self.crop_box_start
        handle = self.active_crop_handle
        if handle == 'move':
            start_x, start_y = self.crop_move_start
            crop_width = right - left
            crop_height = bottom - top
            new_left = max(0.0, min(1.0 - crop_width, left + (event.x - start_x) / width))
            new_top = max(0.0, min(1.0 - crop_height, top + (event.y - start_y) / height))
            self.crop_box = (new_left, new_top, new_left + crop_width, new_top + crop_height)
            self.draw_crop_overlay()
            return
        nx = (max(x, min(x + width, event.x)) - x) / width
        ny = (max(y, min(y + height, event.y)) - y) / height
        if self.is_free_ratio():
            new_left = nx if 'w' in handle else left
            new_right = nx if 'e' in handle else right
            new_top = ny if 'n' in handle else top
            new_bottom = ny if 's' in handle else bottom
            self.crop_box = self.clamp_crop_box((new_left, new_top, new_right, new_bottom))
            self.draw_crop_overlay()
            return
        ratio = self.get_crop_ratio()
        normalized_ratio = self.normalized_ratio(ratio)
        if handle in ('nw', 'ne', 'sw', 'se'):
            anchor_x = right if 'w' in handle else left
            anchor_y = bottom if 'n' in handle else top
            crop_width = max(MIN_CROP, abs(nx - anchor_x), abs(ny - anchor_y) * normalized_ratio)
            crop_height = crop_width / normalized_ratio
            x1, x2 = (anchor_x - crop_width, anchor_x) if 'w' in handle else (anchor_x, anchor_x + crop_width)
            y1, y2 = (anchor_y - crop_height, anchor_y) if 'n' in handle else (anchor_y, anchor_y + crop_height)
            box = (x1, y1, x2, y2)
        elif handle in ('e', 'w'):
            x1, x2 = (left, nx) if handle == 'e' else (nx, right)
            new_height = max(MIN_CROP, x2 - x1) / normalized_ratio
            center_y = (top + bottom) / 2.0
            box = (x1, center_y - new_height / 2.0, x2, center_y + new_height / 2.0)
        else:
            y1, y2 = (top, ny) if handle == 's' else (ny, bottom)
            new_width = max(MIN_CROP, y2 - y1) * normalized_ratio
            center_x = (left + right) / 2.0
            box = (center_x - new_width / 2.0, y1, center_x + new_width / 2.0, y2)
        self.crop_box = self.clamp_fixed_box(box, ratio)
        self.draw_crop_overlay()

    def finish_crop(self, event):
        if self.manual_mode or not self.cropping:
            return
        self.cropping = False
        self.active_crop_handle = None
        self.crop_box_start = None
        self.crop_move_start = None
        box = self.clamp_for_ratio(self.crop_or_full())
        if box[0] <= 0.005 and box[1] <= 0.005 and box[2] >= 0.995 and box[3] >= 0.995:
            self.crop_box = None
        else:
            self.crop_box = box
        self.mark_crop_changed('Crop changed')

    def reset_crop(self):
        if self.input_is_busy(include_manual=False) or self.original_image is None or self.crop_box is None:
            return
        self.crop_box = self.crop_to_ratio(self.ratio_var.get())
        self.mark_crop_changed('Crop reset')

    def draw_crop_overlay(self):
        if self.input_box is None:
            return
        self.in_canvas.delete('crop_overlay')
        self.in_canvas.create_rectangle(*self.crop_display_rect(self.input_box), outline='#000000', width=3, dash=(8, 5), tags='crop_overlay')
        move_x1, move_y1, move_x2, move_y2 = self.move_btn_box(self.input_box)
        self.in_canvas.create_rectangle(move_x1, move_y1, move_x2, move_y2, outline='#000000', fill='#FFFFFF', width=2, tags='crop_overlay')
        self.in_canvas.create_text((move_x1 + move_x2) / 2.0, (move_y1 + move_y2) / 2.0, text='✥', fill='#000000', font=('Segoe UI Symbol', 15), tags='crop_overlay')
        handle_size = self.crop_handle_size()
        for hx, hy in self.crop_handle_points(self.input_box).values():
            self.in_canvas.create_rectangle(hx - handle_size, hy - handle_size, hx + handle_size, hy + handle_size, outline='#000000', fill='#FFFFFF', width=2, tags='crop_overlay')
