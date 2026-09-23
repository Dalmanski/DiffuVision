def center_window(window, width=None, height=None):
    """Center a Tk/Tkinter window on the active screen."""
    window.update_idletasks()

    if width is None:
        width = max(1, window.winfo_width())
    if height is None:
        height = max(1, window.winfo_height())

    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = max(0, (screen_width - width) // 2)
    y = max(0, (screen_height - height) // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")


def center_popup(window, width, height):
    """Center a popup or toplevel window using the given dimensions."""
    center_window(window, width=width, height=height)


"""
from utils.centwin import center_window

center_window(self, width=width, height=height)
"""
