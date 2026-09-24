from pathlib import Path
from tkinter import filedialog


def _initial_save_dir(input_path=None, last_dir=None):
    if last_dir and Path(last_dir).exists():
        return Path(last_dir)
    if input_path:
        return Path(input_path).parent
    return Path.cwd()


def prompt_save_path(title, base_name, input_path=None, last_dir=None):
    initial_dir = _initial_save_dir(input_path=input_path, last_dir=last_dir)
    candidate = initial_dir / f'{base_name}.png'
    counter = 1
    while candidate.exists():
        candidate = initial_dir / f'{base_name} ({counter}).png'
        counter += 1
    path = filedialog.asksaveasfilename(
        title=title,
        initialdir=str(initial_dir),
        initialfile=candidate.name,
        confirmoverwrite=True,
        defaultextension='.png',
        filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg *.jpeg'), ('WebP', '*.webp')],
    )
    if not path:
        return None
    return Path(path)
