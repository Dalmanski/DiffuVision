import json
from pathlib import Path
import customtkinter as ctk
from dotenv import dotenv_values, set_key
from widgets.ctk_theme import configure_ctk_theme
from widgets.centwin import center_window
from widgets.ctk_utils import GradientBtn
from utils.config_manager import ConfigManager

BASE_DIR = Path(__file__).resolve().parent

class ArrayEditor(ctk.CTkFrame):
    def __init__(self, parent, values):
        super().__init__(parent, fg_color="transparent")
        self.values = [str(value) for value in values]
        self.selected = 0
        self.grid_columnconfigure(0, weight=1)
        self.selector = ctk.CTkComboBox(self, values=self.display_values(), command=self.select_value)
        self.selector.grid(row=0, column=0, padx=(0, 6), sticky="ew")
        self.selector.bind("<KeyRelease>", self.update_value)
        self.selector.bind("<FocusOut>", self.refresh)

        self.add_button = GradientBtn(self, text="+", width=34, height=34, command=self.add_value)
        self.add_button.grid(row=0, column=1, padx=3)

        self.remove_button = GradientBtn(self, text="-", width=34, height=34, fg_color="#514f59", hover_color="#3e3c45", command=self.remove_value)
        self.remove_button.grid(row=0, column=2, padx=(3, 0))

        self.refresh()

    def display_values(self):
        return self.values or [""]

    def update_value(self, event=None):
        value = self.selector.get()
        if not value:
            return
        if value in self.values:
            self.selected = self.values.index(value)
        elif self.values and 0 <= self.selected < len(self.values):
            self.values[self.selected] = value

    def refresh(self, event=None):
        current = self.selector.get()
        self.selector.configure(values=self.display_values())
        if self.values:
            if current in self.values:
                self.selected = self.values.index(current)
            else:
                self.selected = max(0, min(self.selected, len(self.values) - 1))
            self.selector.set(self.values[self.selected])
        else:
            self.selected = 0
            self.selector.set("")

    def select_value(self, value):
        self.update_value()
        try:
            self.selected = self.values.index(value)
        except ValueError:
            self.selected = 0
        self.selector.set(self.values[self.selected] if self.values else "")

    def add_value(self):
        self.update_value()
        self.values.append("")
        self.selected = len(self.values) - 1
        self.selector.configure(values=self.display_values())
        self.selector.set("")
        self.selector.focus_set()

    def remove_value(self):
        self.update_value()
        if not self.values:
            return
        current = self.selector.get()
        if current in self.values:
            self.selected = self.values.index(current)
        self.values.pop(self.selected)
        self.selected = max(0, min(self.selected, len(self.values) - 1))
        self.refresh()

    def get_values(self):
        self.update_value()
        return self.values

class DictionaryEditor(ctk.CTkFrame):
    def __init__(self, parent, values):
        super().__init__(parent, fg_color="transparent")
        self.values = {str(key): str(value) for key, value in values.items()}
        self.selected = 0
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.selector = ctk.CTkComboBox(self, values=self.display_values(), command=self.select_value)
        self.selector.grid(row=0, column=0, padx=(0, 6), sticky="ew")
        self.value_entry = ctk.CTkEntry(self)
        self.value_entry.grid(row=0, column=1, padx=(0, 6), sticky="ew")
        self.value_entry.bind("<KeyRelease>", self.update_value)
        self.value_entry.bind("<FocusOut>", self.refresh)
        self.add_button = GradientBtn(self, text="+", width=34, height=34, command=self.add_value)
        self.add_button.grid(row=0, column=2, padx=3)
        self.remove_button = GradientBtn(self, text="-", width=34, height=34, fg_color="#514f59", hover_color="#3e3c45", command=self.remove_value)
        self.remove_button.grid(row=0, column=3, padx=(3, 0))
        self.refresh()

    def display_values(self):
        return list(self.values) or [""]

    def update_value(self, event=None):
        keys = list(self.values)
        if keys and 0 <= self.selected < len(keys):
            self.values[keys[self.selected]] = self.value_entry.get()

    def refresh(self, event=None):
        current = self.selector.get()
        keys = list(self.values)
        self.selector.configure(values=self.display_values())
        if keys:
            self.selected = keys.index(current) if current in keys else max(0, min(self.selected, len(keys) - 1))
            self.selector.set(keys[self.selected])
            self.value_entry.delete(0, "end")
            self.value_entry.insert(0, self.values[keys[self.selected]])
        else:
            self.selected = 0
            self.selector.set("")
            self.value_entry.delete(0, "end")

    def add_value(self):
        self.update_value()
        key = f"key_{len(self.values) + 1}"
        self.values[key] = ""
        self.selected = len(self.values) - 1
        self.refresh()
        self.value_entry.focus_set()

    def remove_value(self):
        self.update_value()
        keys = list(self.values)
        if not keys:
            return
        self.values.pop(keys[self.selected])
        self.selected = max(0, min(self.selected, len(self.values) - 1))
        self.refresh()

    def select_value(self, value):
        self.update_value()
        keys = list(self.values)
        if value in keys:
            self.selected = keys.index(value)
        self.refresh()

    def get_values(self):
        self.update_value()
        return self.values

class BooleanEditor(ctk.CTkOptionMenu):
    def __init__(self, parent, value):
        super().__init__(parent, values=["True", "False"])
        self.set(value)

    def get(self):
        return self.get()

class SettingsPopup(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        window_width = 820
        window_height = 700
        self.title("Settings")
        self.geometry(f"{window_width}x{window_height}")
        self.minsize(650, 500)
        center_window(self, width=window_width, height=window_height)
        self.transient(parent)
        self.grab_set()
        self.entries = {}
        self.files = []
        self.on_save = None
        self.build_ui()
        self.load_env_files()

    def find_env_files(self):
        files = []
        for path in BASE_DIR.rglob(".env*"):
            if path.is_file() and (path.name == ".env" or path.name.startswith(".env.")) and "venv" not in path.parts and ".venv" not in path.parts and "__pycache__" not in path.parts:
                files.append(path)
        return sorted(files)

    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkFrame(self)
        self.header.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        self.header.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(self.header, text="Environment Settings", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=12, pady=12, sticky="w")

        self.refresh_button = GradientBtn(self.header, text="Refresh", command=self.load_env_files)
        self.refresh_button.grid(row=0, column=1, padx=(6, 12), pady=12)

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")
        self.scroll.grid_columnconfigure(0, weight=1)

        self.footer = ctk.CTkFrame(self)
        self.footer.grid(row=2, column=0, padx=12, pady=(6, 12), sticky="ew")
        self.footer.grid_columnconfigure(0, weight=1)

        self.status = ctk.CTkLabel(self.footer, text="Ready", anchor="w")
        self.status.grid(row=0, column=0, padx=12, pady=10, sticky="w")

        self.save_button = GradientBtn(self.footer, text="Save", command=self.save_settings)
        self.save_button.grid(row=0, column=1, padx=6, pady=10)

        self.close_button = GradientBtn(self.footer, text="Close", fg_color="#514f59", hover_color="#3e3c45", command=self.destroy)
        self.close_button.grid(row=0, column=2, padx=(0, 12), pady=10)

    def parse_value(self, value):
        if value is None:
            return "text", ""
        text = value.strip()
        if text.lower() in ("true", "false"):
            return "bool", text.title()
        parsed = ConfigManager.parse_json_dict(text)
        if parsed is not None:
            return "dict", parsed
        parsed = ConfigManager.parse_json_list(text)
        if parsed is not None:
            return "list", parsed
        return "text", value

    def load_env_files(self):
        for widget in self.scroll.winfo_children():
            widget.destroy()

        self.entries.clear()
        self.files = self.find_env_files()

        if not self.files:
            self.status.configure(text="No .env files found")
            empty = ctk.CTkLabel(self.scroll, text="No .env files were found.", text_color="gray65")
            empty.grid(row=0, column=0, padx=12, pady=20)
            return

        row = 0
        total = 0

        for env_file in self.files:
            values = dotenv_values(env_file)
            section = ctk.CTkFrame(self.scroll)
            section.grid(row=row, column=0, padx=4, pady=(4, 10), sticky="ew")
            section.grid_columnconfigure(1, weight=1)

            section_title = ctk.CTkLabel(section, text=str(env_file.relative_to(BASE_DIR)), font=ctk.CTkFont(size=15, weight="bold"))
            section_title.grid(row=0, column=0, columnspan=2, padx=12, pady=(10, 8), sticky="w")

            file_entries = {}
            self.entries[env_file] = file_entries
            section_row = 1

            for key, value in values.items():
                value_type, parsed = self.parse_value(value)
                key_label = ctk.CTkLabel(section, text=key, anchor="w")
                key_label.grid(row=section_row, column=0, padx=(12, 8), pady=5, sticky="w")

                if value_type == "list":
                    widget = ArrayEditor(section, parsed)
                elif value_type == "dict":
                    widget = DictionaryEditor(section, parsed)
                elif value_type == "bool":
                    widget = ctk.CTkOptionMenu(section, values=["True", "False"])
                    widget.set(parsed)
                else:
                    widget = ctk.CTkEntry(section)
                    widget.insert(0, parsed)

                widget.grid(row=section_row, column=1, padx=(0, 12), pady=5, sticky="ew")
                file_entries[key] = (value_type, widget)
                section_row += 1
                total += 1

            row += 1

        self.status.configure(text=f"{len(self.files)} .env file(s) loaded, {total} setting(s)")

    def save_settings(self):
        saved = 0

        for env_file, file_entries in self.entries.items():
            for key, (value_type, widget) in file_entries.items():
                if value_type in {"list", "dict"}:
                    value = json.dumps(widget.get_values(), ensure_ascii=False)
                else:
                    value = widget.get()
                set_key(str(env_file), key, value, quote_mode="auto")
                saved += 1

        self.status.configure(text=f"Saved {saved} setting(s)")
        if callable(self.on_save):
            self.on_save()
        self.destroy()

def open_settings_popup(parent):
    popup = SettingsPopup(parent)
    popup.focus_force()
    return popup

if __name__ == "__main__":
    configure_ctk_theme()
    root = ctk.CTk()
    root.geometry("900x600")
    root.title("App")
    open_button = GradientBtn(root, text="Open Settings", command=lambda: open_settings_popup(root))
    open_button.pack(padx=20, pady=20)
    root.mainloop()