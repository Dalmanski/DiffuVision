import os
from copy import deepcopy
from pathlib import Path

import customtkinter as ctk


def _env_value(name, default=''):
	value = os.getenv(name, default)
	if value is None:
		return default
	value = str(value).strip()
	if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
		value = value[1:-1]
	return value


class ConfigCTkTheme:
	def __init__(self, root=None):
		previous_theme = deepcopy(ctk.ThemeManager.theme)
		themes_dir = Path(__file__).resolve().parent.parent / 'themes'
		mode = _env_value('CTk_mode', 'system').lower()
		theme = _env_value('CTk_theme', 'default.json')
		theme_path = themes_dir / theme
		default_theme_path = themes_dir / 'default.json'

		if not theme_path.is_file():
			theme_path = default_theme_path

		ctk.set_appearance_mode(mode if mode in {'system', 'light', 'dark'} else 'system')
		ctk.set_default_color_theme(str(theme_path))

		if root is not None:
			from widgets.ctk_utils import CanvasCTk, GradientBtn
			current_theme = ctk.ThemeManager.theme

			def color(widget, name):
				try:
					return widget.cget(name)
				except Exception:
					return None

			def refresh(widget, old_parent_bg=None, new_parent_bg=None):
				if isinstance(widget, CanvasCTk):
					widget.refresh_theme()
				elif isinstance(widget, GradientBtn):
					widget.refresh_theme()
					return

				old_bg = color(widget, 'bg_color')
				old_fg = color(widget, 'fg_color')
				inherits_parent_bg = old_parent_bg is not None and old_bg in (old_parent_bg, new_parent_bg)
				if inherits_parent_bg and old_bg == old_parent_bg and old_parent_bg != new_parent_bg:
					try:
						widget.configure(bg_color=new_parent_bg)
					except Exception:
						inherits_parent_bg = False

				section = next((base.__name__ for base in type(widget).__mro__ if base.__name__ in current_theme and base.__name__.startswith('CTk')), None)
				scrollable_fill_updated = False
				if section:
					old_options = previous_theme.get(section, {})
					for key, value in current_theme[section].items():
						if key not in old_options or old_options[key] == value:
							continue
						try:
							current_value = widget.cget(key)
							uses_old_default = current_value == old_options[key]
							if section == 'CTkScrollableFrame' and key == 'fg_color':
								frame_old_color = previous_theme.get('CTkFrame', {}).get(key)
								frame_new_color = current_theme.get('CTkFrame', {}).get(key)
								uses_old_default = uses_old_default or current_value in (frame_old_color, frame_new_color)
							if uses_old_default:
								widget.configure(**{key: value})
								if section == 'CTkScrollableFrame' and key == 'fg_color':
									scrollable_fill_updated = True
						except Exception:
							continue

				new_bg = color(widget, 'bg_color')
				new_fg = color(widget, 'fg_color')
				old_surface = old_parent_bg if inherits_parent_bg else old_bg
				new_surface = new_parent_bg if inherits_parent_bg else new_bg
				if old_fg != 'transparent':
					old_surface = old_fg
				if new_fg != 'transparent':
					new_surface = new_fg
				if scrollable_fill_updated:
					old_surface = previous_theme.get('CTkFrame', {}).get('fg_color', old_surface)
					new_surface = current_theme[section]['fg_color']
				for child in widget.winfo_children():
					refresh(child, old_surface, new_surface)

			refresh(root)
