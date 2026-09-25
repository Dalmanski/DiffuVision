import ast
import customtkinter as ctk
from PIL import Image, ImageDraw

def font(family=None, size=None, weight=None):
    theme = ctk.ThemeManager.theme.get('CTkFont', {})
    family = family or theme.get('family', 'Arial')
    size = size or theme.get('size', 13)
    weight = weight or theme.get('weight', 'normal')
    return ctk.CTkFont(family=family, size=size, weight=weight)

def _theme_value(key, fallback):
    theme = ctk.ThemeManager.theme.get('CTkButton', {})
    value = theme.get(key, fallback)
    return _resolve_color(value)

def _theme_color(key):
    return _theme_value(key, ['#DCE3EA', '#202A36'])

def _resolve_color(value):
    if isinstance(value, str) and value.startswith('['):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            pass
    if isinstance(value, list):
        return value[1] if ctk.get_appearance_mode().lower() == 'dark' else value[0]
    return value

def _rgb(color):
    color = _resolve_color(color)
    color = str(color).lstrip('#')
    if len(color) != 6:
        return 0, 0, 0
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

def _darken(color, amount=0.28):
    return '#%02x%02x%02x' % tuple(max(0, int(channel * (1 - amount))) for channel in _rgb(color))

class GradientBtn(ctk.CTkFrame):
    def __init__(self, master, text='', command=None, **kwargs):
        self._text_anchor = kwargs.pop('anchor', 'center')
        self._uses_theme_corner_radius = 'corner_radius' not in kwargs
        self._uses_theme_border_width = 'border_width' not in kwargs
        self._uses_theme_border_color = 'border_color' not in kwargs
        self._corner_radius = kwargs.pop('corner_radius', _theme_value('corner_radius', 6))
        self._border_width = kwargs.pop('border_width', _theme_value('border_width', 0))
        self._border_color = _resolve_color(kwargs.pop('border_color', _theme_color('border_color')))
        self._command = command
        self._state = kwargs.pop('state', 'normal')
        self._text = text
        self._font = kwargs.pop('font', font(weight='bold'))
        self._uses_theme_text_color = 'text_color' not in kwargs
        self._uses_theme_fg_color = 'fg_color' not in kwargs
        self._uses_theme_hover_color = 'hover_color' not in kwargs
        self._text_color = _resolve_color(kwargs.pop('text_color', _theme_color('text_color')))
        self._text_color_disabled = _resolve_color(_theme_color('text_color_disabled'))
        self._start = _resolve_color(kwargs.pop('fg_color', _theme_color('fg_color')))
        self._end = _resolve_color(kwargs.pop('hover_color', _darken(self._start)))
        self._hovered = False
        kwargs.setdefault('width', ctk.ThemeManager.theme.get('CTkButton', {}).get('width', 120))
        kwargs.setdefault('height', ctk.ThemeManager.theme.get('CTkButton', {}).get('height', 28))
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
        if self._border_width > 0:
            draw = ImageDraw.Draw(image)
            border = _rgb(self._border_color)
            draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=self._corner_radius * scaling, outline=border + (255,), width=max(1, int(self._border_width * scaling)))
        self._image = ctk.CTkImage(light_image=image, dark_image=image, size=(width / scaling, height / scaling))
        text_color = self._text_color if self._state == 'normal' else self._text_color_disabled
        self._label.configure(image=self._image, text=self._text, text_color=text_color, font=self._font)
        self._place_label()

    def refresh_theme(self):
        if self._uses_theme_text_color:
            self._text_color = _theme_color('text_color')
        if self._uses_theme_fg_color:
            self._start = _theme_color('fg_color')
        if self._uses_theme_hover_color:
            self._end = _darken(self._start)
        if self._uses_theme_border_color:
            self._border_color = _theme_color('border_color')
        if self._uses_theme_corner_radius:
            self._corner_radius = _theme_value('corner_radius', 6)
        if self._uses_theme_border_width:
            self._border_width = _theme_value('border_width', 0)
        self._text_color_disabled = _resolve_color(_theme_color('text_color_disabled'))
        self._render()

    def configure(self, **kwargs):
        button_keys = {'text', 'state', 'command', 'fg_color', 'hover_color', 'text_color', 'font', 'corner_radius', 'border_width', 'border_color', 'anchor'}
        button_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in button_keys}
        if 'text' in button_kwargs:
            self._text = button_kwargs['text']
        if 'state' in button_kwargs:
            self._state = button_kwargs['state']
        if 'command' in button_kwargs:
            self._command = button_kwargs['command']
        if 'fg_color' in button_kwargs:
            self._uses_theme_fg_color = False
            self._start = _resolve_color(button_kwargs['fg_color'])
            self._end = _darken(self._start)
        if 'hover_color' in button_kwargs:
            self._uses_theme_hover_color = False
            self._end = _resolve_color(button_kwargs['hover_color'])
        if 'text_color' in button_kwargs:
            self._uses_theme_text_color = False
            self._text_color = _resolve_color(button_kwargs['text_color'])
        if 'font' in button_kwargs:
            self._font = button_kwargs['font']
        if 'corner_radius' in button_kwargs:
            self._uses_theme_corner_radius = False
            self._corner_radius = button_kwargs['corner_radius']
        if 'border_width' in button_kwargs:
            self._uses_theme_border_width = False
            self._border_width = button_kwargs['border_width']
        if 'border_color' in button_kwargs:
            self._uses_theme_border_color = False
            self._border_color = _resolve_color(button_kwargs['border_color'])
        if 'anchor' in button_kwargs:
            self._text_anchor = button_kwargs['anchor']
            self._place_label()
        if button_kwargs:
            self._render()
        return super().configure(**kwargs)