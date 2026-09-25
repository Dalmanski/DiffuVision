import sys

import customtkinter as ctk

class ConsoleTextBox(ctk.CTkTextbox):
    def __init__(self, parent, height=260, wrap='none', font=('Consolas', 12), fg_color='#000000', text_color='#D0D0D0', **kwargs):
        super().__init__(parent, height=height, wrap=wrap, font=font, fg_color=fg_color, text_color=text_color, **kwargs)
        self.configure(state='disabled')

    def _render_log(self, text, live):
        try:
            self.configure(state='normal')
            if live:
                line_start = self.index('end-1c linestart')
                self.delete(line_start, 'end-1c')
                self.insert('end', text)
            else:
                self.insert('end', text)
                if not text.endswith('\n'):
                    self.insert('end', '\n')
            self.see('end')
            self.configure(state='disabled')
        except Exception:
            pass

    def log(self, text, color=None, live=False):
        if text is None or '[DEBUG]' in str(text):
            return
        text = str(text).replace('\x1b[K', '').replace('\x1b[2K', '').strip('\n')
        if not text:
            return
        try:
            self.after(0, self._render_log, text, live)
        except Exception:
            pass

    def clear(self):
        try:
            self.configure(state='normal')
            self.delete('1.0', 'end')
            self.configure(state='disabled')
        except Exception:
            pass

class ConsoleRedirect:
    def __init__(self, console, original=None, show_in_console=True):
        self.console = console
        self.original = original
        self.show_in_console = show_in_console
        self.buffer = ""

    def write(self, text):
        value = str(text)
        if self.original is not None:
            try:
                self.original.write(value)
            except Exception:
                pass
        if not self.show_in_console or not value or '[DEBUG]' in value:
            return len(value)
        if '\r' in value:
            parts = value.split('\r')
            value = parts[-1]
            self.console.log(value, live=True)
            return len(str(text))
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                try:
                    self.console.log(line)
                except Exception:
                    pass
        return len(value)

    def flush(self):
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass
        if self.show_in_console and self.buffer.strip():
            try:
                self.console.log(self.buffer.rstrip())
            except Exception:
                pass
            self.buffer = ""

    def isatty(self):
        if self.original is None:
            return False
        try:
            return self.original.isatty()
        except Exception:
            return False

def create_redirects(console):
    return ConsoleRedirect(console, original=sys.stdout, show_in_console=True), ConsoleRedirect(console, original=sys.stderr, show_in_console=False)