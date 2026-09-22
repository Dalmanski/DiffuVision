import os
from pathlib import Path

import customtkinter as ctk


def configure_ctk_theme():
	themes_dir = Path(__file__).resolve().parent.parent / 'themes'
	mode = str(os.getenv('CTk_mode') or 'system').strip().lower()
	theme = str(os.getenv('CTk_theme') or 'default.json').strip()
	theme_path = themes_dir / theme
	default_theme_path = themes_dir / 'default.json'

	if not theme_path.is_file():
		theme_path = default_theme_path

	ctk.set_appearance_mode(mode if mode in {'system', 'light', 'dark'} else 'system')
	ctk.set_default_color_theme(str(theme_path))

'''
from utils.ctk_theme import configure_ctk_theme

configure_ctk_theme()
'''
