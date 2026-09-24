import os
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


def configure_ctk_theme():
	themes_dir = Path(__file__).resolve().parent.parent / 'themes'
	mode = _env_value('CTk_mode', 'system').lower()
	theme = _env_value('CTk_theme', 'default.json')
	theme_path = themes_dir / theme
	default_theme_path = themes_dir / 'default.json'

	if not theme_path.is_file():
		theme_path = default_theme_path

	ctk.set_appearance_mode(mode if mode in {'system', 'light', 'dark'} else 'system')
	ctk.set_default_color_theme(str(theme_path))

'''
from widgets.ctk_theme import configure_ctk_theme

configure_ctk_theme()
'''
