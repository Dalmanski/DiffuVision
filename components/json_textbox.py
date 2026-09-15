import json
import re
import tkinter as tk
import customtkinter as ctk

class JSONTextBox(ctk.CTkFrame):
    def __init__(self, master, height=220, font_size=12, fg_color='#000000'):
        super().__init__(master, fg_color=fg_color)
        self.change_callback = None
        self.grid_propagate(False)
        self.pack_propagate(False)
        self.text = tk.Text(self, wrap='none', undo=True, bg='#000000', fg='#D4D4D4', insertbackground='#AEAFAD', selectbackground='#264F78', selectforeground='#FFFFFF', relief='flat', borderwidth=0, highlightthickness=0, font=('Consolas', font_size), height=max(1, int(height / 17)))
        self.text.grid(row=0, column=0, sticky='nsew')
        self.y_scroll = ctk.CTkScrollbar(self, orientation='vertical', command=self.text.yview)
        self.y_scroll.grid(row=0, column=1, sticky='ns')
        self.x_scroll = ctk.CTkScrollbar(self, orientation='horizontal', command=self.text.xview)
        self.x_scroll.grid(row=1, column=0, sticky='ew')
        self.text.configure(yscrollcommand=self.y_scroll.set, xscrollcommand=self.x_scroll.set)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.text.tag_configure('key', foreground='#9CDCFE')
        self.text.tag_configure('string', foreground='#CE9178')
        self.text.tag_configure('number', foreground='#B5CEA8')
        self.text.tag_configure('boolean', foreground='#569CD6')
        self.text.tag_configure('null', foreground='#569CD6')
        self.text.tag_configure('punctuation', foreground='#D4D4D4')
        self.text.tag_configure('brace', foreground='#D4D4D4')
        self.text.tag_configure('invalid', foreground='#F44747')
        self.text.bind('<<Modified>>', self.on_change)
        self.text.bind('<<Paste>>', self.on_paste)
        self.text.bind('<<Cut>>', self.on_cut)
        self.text.edit_modified(False)

    def set_change_callback(self, callback):
        self.change_callback = callback

    def on_change(self, event=None):
        if not self.text.edit_modified():
            return
        self.text.edit_modified(False)
        self.highlight()
        if self.change_callback:
            self.change_callback(event)

    def on_paste(self, event=None):
        self.after(10, self.highlight)

    def on_cut(self, event=None):
        self.after(10, self.highlight)

    def get(self, start='1.0', end='end'):
        return self.text.get(start, end)

    def delete(self, start, end=None):
        self.text.delete(start, end)
        self.highlight()

    def insert(self, index, chars, *tags):
        self.text.insert(index, chars, *tags)
        self.highlight()

    def focus_set(self):
        self.text.focus_set()

    def highlight(self):
        content = self.text.get('1.0', 'end-1c')
        for tag in ('key', 'string', 'number', 'boolean', 'null', 'punctuation', 'brace', 'invalid'):
            self.text.tag_remove(tag, '1.0', 'end')
        token_pattern = re.compile(r'"(?:\\.|[^"\\])*"|\b(?:true|false|null)\b|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|[{}\[\],:]')
        for match in token_pattern.finditer(content):
            token = match.group(0)
            start = f'1.0+{match.start()}c'
            end = f'1.0+{match.end()}c'
            if token.startswith('"'):
                rest = content[match.end():]
                after = rest.lstrip()
                is_key = after.startswith(':')
                self.text.tag_add('key' if is_key else 'string', start, end)
            elif token in ('true', 'false'):
                self.text.tag_add('boolean', start, end)
            elif token == 'null':
                self.text.tag_add('null', start, end)
            elif token[0].isdigit() or token[0] == '-':
                self.text.tag_add('number', start, end)
            elif token in '{}[]':
                self.text.tag_add('brace', start, end)
            else:
                self.text.tag_add('punctuation', start, end)
        try:
            json.loads(content)
        except json.JSONDecodeError as error:
            position = max(0, min(error.pos, len(content)))
            start = f'1.0+{position}c'
            end = f'1.0+{min(position + 1, len(content))}c'
            if position < len(content):
                self.text.tag_add('invalid', start, end)
