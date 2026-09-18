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
    def __init__(self, console):
        self.console = console

    def write(self, text):
        if not text or '[DEBUG]' in text:
            return
        if '\r' in text:
            parts = text.split('\r')
            if len(parts) > 1:
                text = parts[-1]
            self.console.log(text, live=True)
        else:
            self.console.log(text)

    def flush(self):
        pass

def create_redirects(console):
    return ConsoleRedirect(console), ConsoleRedirect(console)