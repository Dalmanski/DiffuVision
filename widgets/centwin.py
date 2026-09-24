
def center_window(window, width=None, height=None):
    window.update_idletasks()
    width = width or window.winfo_reqwidth()
    height = height or window.winfo_reqheight()
    window.geometry(f"{width}x{height}+0+0")
    window.update()
    border = window.winfo_rootx() - window.winfo_x()
    titlebar = window.winfo_rooty() - window.winfo_y()
    x = max(0, (window.winfo_screenwidth() - width - 2 * border) // 2)
    y = max(0, (window.winfo_screenheight() - height - titlebar - border) // 2)
    window.geometry(f"+{x}+{y}")


def center_popup(window, width, height):
    center_window(window, width=width, height=height)

"""
from utils.centwin import center_window

center_window(self, width=67, height=67)
"""
