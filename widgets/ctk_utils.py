import customtkinter as ctk
from PIL import Image, ImageDraw

FONT_FAMILY = ctk.ThemeManager.theme['CTkFont']['family']
FONT_SIZE = ctk.ThemeManager.theme['CTkFont']['size']
FONT_WEIGHT = ctk.ThemeManager.theme['CTkFont']['weight']
BTN_CORNER_RAD = ctk.ThemeManager.theme['CTkButton']['corner_radius']
BTN_WIDTH = ctk.ThemeManager.theme['CTkButton'].get('width', 120)
BTN_HEIGHT = ctk.ThemeManager.theme['CTkButton'].get('height', 28)

def font(family=FONT_FAMILY, size=FONT_SIZE, weight=FONT_WEIGHT):
    return ctk.CTkFont(family=family, size=size, weight=weight)

def _theme_color(key):
    theme = ctk.ThemeManager.theme.get('CTkButton', {})
    value = theme.get(key, ['#DCE3EA', '#202A36'])
    return _resolve_color(value)

def _resolve_color(value):
    if isinstance(value, list):
        return value[1] if ctk.get_appearance_mode().lower() == 'dark' else value[0]
    return value

def _rgb(color):
    color = str(color).lstrip('#')
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

def _darken(color, amount=0.28):
    return '#%02x%02x%02x' % tuple(max(0, int(channel * (1 - amount))) for channel in _rgb(color))

class GradientBtn(ctk.CTkFrame):
    def __init__(self, master, text='', command=None, **kwargs):
        self._text_anchor = kwargs.pop('anchor', 'center')
        self._corner_radius = kwargs.setdefault('corner_radius', BTN_CORNER_RAD)
        self._command = command
        self._state = kwargs.pop('state', 'normal')
        self._text = text
        self._font = kwargs.pop('font', font(weight='bold'))
        self._text_color = kwargs.pop('text_color', _theme_color('text_color'))
        self._start = kwargs.pop('fg_color', _theme_color('fg_color'))
        self._end = kwargs.pop('hover_color', _darken(self._start))
        self._hovered = False
        kwargs.setdefault('width', BTN_WIDTH)
        kwargs.setdefault('height', BTN_HEIGHT)
        kwargs.setdefault('fg_color', 'transparent')
        super().__init__(master, **kwargs)
        self._label = ctk.CTkLabel(self, text=self._text, fg_color='transparent', text_color=self._text_color, font=self._font, compound='center', corner_radius=0)
        self._place_label()
        self.pack_propagate(False)
        for widget in (self, self._label):
            widget.bind('<Configure>', self._resize)
            widget.bind('<Enter>', self._enter)
            widget.bind('<Leave>', self._leave)
            widget.bind('<Button-1>', self._click)
        self._render()

    def _resize(self, event=None):
        if self.winfo_width() > 1 and self.winfo_height() > 1:
            self._render()

    def _enter(self, event=None):
        self._hovered = True
        self._render()

    def _leave(self, event=None):
        self._hovered = False
        self._render()

    def _click(self, event=None):
        if self._state == 'normal' and self._command is not None:
            self._command()

    def _place_label(self):
        self._label.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._label.configure(anchor='w' if self._text_anchor == 'w' else 'center')

    def _mask(self, width, height, radius):
        scale = 4
        mask = Image.new('L', (width * scale, height * scale), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, width * scale - 1, height * scale - 1), radius=radius * scale, fill=255)
        return mask.resize((width, height), Image.LANCZOS)

    def _render(self):
        amount = 0.12 if self._hovered else 0
        start_rgb = _rgb(_darken(self._start, amount))
        end_rgb = _rgb(_darken(self._end, amount))
        scaling = self._get_widget_scaling()
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        image = Image.new('RGBA', (width, height))
        draw = ImageDraw.Draw(image)
        for y in range(height):
            ratio = y / max(height - 1, 1)
            draw.line((0, y, width, y), fill=tuple(int(start_rgb[i] + (end_rgb[i] - start_rgb[i]) * ratio) for i in range(3)) + (255,))
        image.putalpha(self._mask(width, height, self._corner_radius * scaling))
        self._image = ctk.CTkImage(light_image=image, dark_image=image, size=(width / scaling, height / scaling))
        self._label.configure(image=self._image, text=self._text, text_color=self._text_color, font=self._font)
        self._place_label()

    def configure(self, **kwargs):
        button_keys = {'text', 'state', 'command', 'fg_color', 'hover_color', 'text_color', 'font', 'corner_radius', 'anchor'}
        button_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in button_keys}
        if 'text' in button_kwargs:
            self._text = button_kwargs['text']
        if 'state' in button_kwargs:
            self._state = button_kwargs['state']
        if 'command' in button_kwargs:
            self._command = button_kwargs['command']
        if 'fg_color' in button_kwargs:
            self._start = _resolve_color(button_kwargs['fg_color'])
            self._end = _darken(self._start)
        if 'hover_color' in button_kwargs:
            self._end = _resolve_color(button_kwargs['hover_color'])
        if 'text_color' in button_kwargs:
            self._text_color = _resolve_color(button_kwargs['text_color'])
        if 'font' in button_kwargs:
            self._font = button_kwargs['font']
        if 'corner_radius' in button_kwargs:
            self._corner_radius = button_kwargs['corner_radius']
        if 'anchor' in button_kwargs:
            self._text_anchor = button_kwargs['anchor']
            self._place_label()
        if button_kwargs:
            self._render()
        return super().configure(**kwargs)