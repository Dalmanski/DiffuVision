import math
from PIL import Image

class CropImageMixin:
    def set_recommended_ratio(self, image):
        width, height = image.size
        aspect = width / height
        ratio_sizes = self.recommended_ratio_sizes
        choice = min(ratio_sizes, key=lambda key: abs(math.log(aspect / (ratio_sizes[key][0] / ratio_sizes[key][1]))))
        self.ratio_var.set(choice)
        self.crop_box = self.default_crop_box_for_ratio(choice)

    def default_crop_box_for_ratio(self, choice):
        if self.original_image is None:
            return 0.0, 0.0, 1.0, 1.0
        ratio = float(choice.split(':')[0]) / float(choice.split(':')[1])
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

    def get_effective_crop_box(self):
        if self.crop_box is None:
            return 0.0, 0.0, 1.0, 1.0
        return self.crop_box

    def get_cropped_image(self, image):
        image = image.convert('RGB')
        if self.crop_box is None:
            return image.copy()
        width, height = image.size
        left, top, right, bottom = self.crop_box
        x1 = max(0, min(width - 1, int(round(left * width))))
        y1 = max(0, min(height - 1, int(round(top * height))))
        x2 = max(x1 + 1, min(width, int(round(right * width))))
        y2 = max(y1 + 1, min(height, int(round(bottom * height))))
        return image.crop((x1, y1, x2, y2))

    def clamp_crop_box(self, box):
        left, top, right, bottom = [float(value) for value in box]
        minimum = 0.02
        left = max(0.0, min(1.0 - minimum, left))
        top = max(0.0, min(1.0 - minimum, top))
        right = max(minimum, min(1.0, right))
        bottom = max(minimum, min(1.0, bottom))
        if right - left < minimum:
            if left + minimum <= 1.0:
                right = left + minimum
            else:
                left = max(0.0, right - minimum)
        if bottom - top < minimum:
            if top + minimum <= 1.0:
                bottom = top + minimum
            else:
                top = max(0.0, bottom - minimum)
        return left, top, right, bottom

    def get_crop_ratio(self):
        value = self.ratio_var.get().strip().upper()
        if value == 'FREE':
            if self.original_image is None:
                return None
            left, top, right, bottom = self.get_effective_crop_box()
            crop_width = max(1.0, (right - left) * self.original_image.width)
            crop_height = max(1.0, (bottom - top) * self.original_image.height)
            return crop_width / crop_height
        left, right = value.split(':', 1)
        return float(left) / float(right)

    def clamp_fixed_crop_box(self, box, ratio):
        if self.original_image is None or ratio is None:
            return self.clamp_crop_box(box)
        left, top, right, bottom = [float(value) for value in box]
        normalized_ratio = ratio * self.original_image.height / self.original_image.width
        width = max(0.02, right - left)
        height = max(0.02, bottom - top)
        if width / height > normalized_ratio:
            height = width / normalized_ratio
        else:
            width = height * normalized_ratio
        width = min(width, 1.0, normalized_ratio)
        height = min(height, 1.0, 1.0 / normalized_ratio)
        if width < 0.02:
            width = 0.02
            height = width / normalized_ratio
        if height < 0.02:
            height = 0.02
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

    def mark_crop_changed(self, message):
        self.reset_preview_state()
        self.refresh_crop_related_ui(allow_reload=True)
        print(message)
        self.show_input()
        self.show_output()

    def ratio_changed(self, choice):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        if self.original_image is None:
            return
        if choice.strip().upper() == 'FREE':
            self.crop_box = 0.0, 0.0, 1.0, 1.0
            self.mark_crop_changed(f'Ratio {choice}')
            return
        ratio = self.get_crop_ratio()
        self.crop_box = self.clamp_fixed_crop_box(self.get_effective_crop_box(), ratio)
        self.mark_crop_changed(f'Ratio {choice}')

    def crop_handle_size(self):
        return 11

    def crop_handle_points(self, display):
        x, y, width, height = display
        left, top, right, bottom = self.get_effective_crop_box()
        x1 = x + left * width
        y1 = y + top * height
        x2 = x + right * width
        y2 = y + bottom * height
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        return {'nw': (x1, y1), 'n': (xm, y1), 'ne': (x2, y1), 'w': (x1, ym), 'e': (x2, ym), 'sw': (x1, y2), 's': (xm, y2), 'se': (x2, y2)}

    def get_crop_handle(self, event):
        if self.input_display_info is None:
            return None
        size = self.crop_handle_size() + 5
        for handle, point in self.crop_handle_points(self.input_display_info).items():
            if abs(event.x - point[0]) <= size and abs(event.y - point[1]) <= size:
                return handle
        return None

    def crop_move_button_size(self):
        return 34, 28

    def crop_move_button_box(self, display):
        x, y, width, height = display
        left, top, right, bottom = self.get_effective_crop_box()
        x2 = x + right * width
        y1 = y + top * height
        button_width, button_height = self.crop_move_button_size()
        return x2 - button_width, y1, x2, y1 + button_height

    def get_crop_move_button(self, event):
        if self.input_display_info is None:
            return False
        x1, y1, x2, y2 = self.crop_move_button_box(self.input_display_info)
        return x1 <= event.x <= x2 and y1 <= event.y <= y2

    def start_crop(self, event):
        if self.manual_segment_mode:
            self.select_manual_segment(event)
            return
        if self.original_image is None or self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        if self.input_display_info is None:
            return
        if self.get_crop_move_button(event):
            self.active_crop_handle = 'move'
            self.crop_box_start = self.get_effective_crop_box()
            self.crop_move_start = (event.x, event.y)
            self.cropping = True
            self.draw_crop_overlay()
            return
        handle = self.get_crop_handle(event)
        if handle is None:
            return
        self.active_crop_handle = handle
        self.crop_box_start = self.get_effective_crop_box()
        self.cropping = True

    def update_crop_selection(self, event):
        if self.manual_segment_mode:
            return
        if not self.cropping or self.input_display_info is None or self.crop_box_start is None:
            return
        if self.active_crop_handle == 'move':
            x, y, width, height = self.input_display_info
            left, top, right, bottom = self.crop_box_start
            start_x, start_y = self.crop_move_start
            crop_width = right - left
            crop_height = bottom - top
            dx = (event.x - start_x) / width
            dy = (event.y - start_y) / height
            new_left = max(0.0, min(1.0 - crop_width, left + dx))
            new_top = max(0.0, min(1.0 - crop_height, top + dy))
            self.crop_box = (new_left, new_top, new_left + crop_width, new_top + crop_height)
            self.draw_crop_overlay()
            return
        x, y, width, height = self.input_display_info
        px = max(x, min(x + width, event.x))
        py = max(y, min(y + height, event.y))
        nx = (px - x) / width
        ny = (py - y) / height
        left, top, right, bottom = self.crop_box_start
        ratio = self.get_crop_ratio()
        normalized_ratio = ratio * self.original_image.height / self.original_image.width
        if self.active_crop_handle in ('nw', 'ne', 'sw', 'se'):
            anchor_x = right if self.active_crop_handle in ('nw', 'sw') else left
            anchor_y = bottom if self.active_crop_handle in ('nw', 'ne') else top
            dx = abs(nx - anchor_x)
            dy = abs(ny - anchor_y)
            crop_width = max(0.02, dx, dy * normalized_ratio)
            crop_height = crop_width / normalized_ratio
            if self.active_crop_handle == 'nw':
                box = (anchor_x - crop_width, anchor_y - crop_height, anchor_x, anchor_y)
            elif self.active_crop_handle == 'ne':
                box = (anchor_x, anchor_y - crop_height, anchor_x + crop_width, anchor_y)
            elif self.active_crop_handle == 'sw':
                box = (anchor_x - crop_width, anchor_y, anchor_x, anchor_y + crop_height)
            else:
                box = (anchor_x, anchor_y, anchor_x + crop_width, anchor_y + crop_height)
        elif self.active_crop_handle in ('e', 'w'):
            if self.active_crop_handle == 'e':
                new_width = max(0.02, nx - left)
                center_y = (top + bottom) / 2.0
                new_height = new_width / normalized_ratio
                box = (left, center_y - new_height / 2.0, nx, center_y + new_height / 2.0)
            else:
                new_width = max(0.02, right - nx)
                center_y = (top + bottom) / 2.0
                new_height = new_width / normalized_ratio
                box = (nx, center_y - new_height / 2.0, right, center_y + new_height / 2.0)
        elif self.active_crop_handle == 's':
            new_height = max(0.02, ny - top)
            center_x = (left + right) / 2.0
            new_width = new_height * normalized_ratio
            box = (center_x - new_width / 2.0, top, center_x + new_width / 2.0, ny)
        else:
            new_height = max(0.02, bottom - ny)
            center_x = (left + right) / 2.0
            new_width = new_height * normalized_ratio
            box = (center_x - new_width / 2.0, ny, center_x + new_width / 2.0, bottom)
        self.crop_box = self.clamp_fixed_crop_box(box, ratio)
        self.draw_crop_overlay()

    def finish_crop(self, event):
        if self.manual_segment_mode:
            return
        if not self.cropping:
            return
        self.cropping = False
        self.active_crop_handle = None
        self.crop_box_start = None
        self.crop_move_start = None
        box = self.clamp_fixed_crop_box(self.crop_box if self.crop_box is not None else (0.0, 0.0, 1.0, 1.0), self.get_crop_ratio())
        if box[0] <= 0.005 and box[1] <= 0.005 and box[2] >= 0.995 and box[3] >= 0.995:
            self.crop_box = None
        else:
            self.crop_box = box
        self.reset_preview_state()
        self.refresh_crop_related_ui(allow_reload=True)
        print('Crop changed')
        self.show_input()
        self.show_output()

    def reset_crop(self):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        if self.original_image is None:
            return
        if self.crop_box is None:
            return
        self.crop_box = self.default_crop_box_for_ratio(self.ratio_var.get())
        self.reset_preview_state()
        self.refresh_crop_related_ui(allow_reload=True)
        print('Crop reset')
        self.show_input()
        self.show_output()

    def draw_crop_overlay(self):
        if self.input_display_info is None:
            return
        self.input_canvas.delete('crop_overlay')
        x, y, width, height = self.input_display_info
        left, top, right, bottom = self.get_effective_crop_box()
        x1 = x + left * width
        y1 = y + top * height
        x2 = x + right * width
        y2 = y + bottom * height
        self.input_canvas.create_rectangle(x1, y1, x2, y2, outline='#000000', width=3, dash=(8, 5), tags='crop_overlay')
        move_x1, move_y1, move_x2, move_y2 = self.crop_move_button_box(self.input_display_info)
        self.input_canvas.create_rectangle(move_x1, move_y1, move_x2, move_y2, outline='#000000', fill='#FFFFFF', width=2, tags='crop_overlay')
        self.input_canvas.create_text((move_x1 + move_x2) / 2.0, (move_y1 + move_y2) / 2.0, text='✥', fill='#000000', font=('Segoe UI Symbol', 15), tags='crop_overlay')
        handle_size = self.crop_handle_size()
        for point in self.crop_handle_points(self.input_display_info).values():
            hx, hy = point
            self.input_canvas.create_rectangle(hx - handle_size, hy - handle_size, hx + handle_size, hy + handle_size, outline='#000000', fill='#FFFFFF', width=2, tags='crop_overlay')