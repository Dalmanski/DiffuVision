import os, sys, json, time, threading, gc, subprocess
from pathlib import Path
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageEnhance, ImageFilter, ImageTk, ImageOps
import numpy as np, torch
import torchvision.transforms.functional as TF
sys.modules.setdefault('torchvision.transforms.functional_tensor', TF)
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from modules.img_classify import AGE_CLASS, GENDER_CLASS, IMAGE_CLASS, classify_image, predict_age, predict_gender
from modules.upscale_img import enhance
from modules.segment_img import SegmentImageMixin
from widgets.json_textbox import JSONTextBox
from widgets.console_textbox import ConsoleTextBox, create_redirects
from widgets.ctk_theme import configure_ctk_theme
from widgets.crop_img import CropImageMixin
from widgets.ctk_utils import GradientBtn
from widgets.gif_anim import GifAnimationMixin
from utils.config_manager import ConfigManager
from utils.access_gate import open_payload
from utils.image_loader import load_image
from utils.dupfilename import prompt_save_path
from modules.sd_ideal import SDIdealImageMixin, OUTPUT_TARGET, RECOMMENDED_RATIO_SIZES

BASE_DIR = Path(__file__).resolve().parent
ConfigManager.load_env(BASE_DIR)
configure_ctk_theme()
MODEL_DIR = BASE_DIR / 'model'
DEFAULT_JSON = BASE_DIR / 'config/sd/default.json'
CHILI_BIN = BASE_DIR / 'config/sd/chili.bin'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_OPTIONS = {}
DEFAULT_MODEL = ''
RATIO_OPTIONS = ['FREE', '1:1', '4:3', '3:2', '16:9', '5:4', '4:5', '3:4', '2:3', '9:16']
LOADING_GIF = 'images\\gif\\loading.gif'

class GenerationStopped(Exception):
    pass

class App(GifAnimationMixin, SegmentImageMixin, CropImageMixin, SDIdealImageMixin, ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('DiffuVision - Inpainting with Stable Diffusion')
        self.iconbitmap('favicon.ico')
        self.geometry('1600x1150')
        self.minsize(1100, 900)
        self.pipe = None
        self.model_name = None
        self.dtype = None
        self.segmentation = None
        self.segmenter = None
        self.sel_img = None
        self.manual_mode = False
        self.manual_loading = False
        self.manual_segmenter = None
        self.manual_selection_image = None
        self.manual_base_mask = None
        self.original_image = None
        self.input_image = None
        self.sd_input_image = None
        self.output_image = None
        self.input_path = None
        self.img_name = None
        self.last_dir = None
        self.gen_count = 0
        self.mask_image = None
        self.mask_src = None
        self.cls_res = None
        self.gender_res = None
        self.age_res = None
        self.class_loading = False
        self.processing = False
        self.models_ready = False
        self.model_loading = False
        self.seg_loading = False
        self.start_time = 0
        self.save_job = None
        self.input_pic = None
        self.output_pic = None
        self.input_busy_visual = False
        self.crop_box = None
        self.crop_start = None
        self.input_box = None
        self.cropping = False
        self.crop_handle = None
        self.sd_rec = ctk.BooleanVar(value=True)
        self.esrgan_out = ctk.BooleanVar(value=False)
        self.full_out = ctk.BooleanVar(value=False)
        self.apply_meta = ctk.BooleanVar(value=True)
        self.stop_requested = threading.Event()
        self.closing = False
        self.autosave = ctk.BooleanVar(value=True)
        self.model_var = ctk.StringVar(value='')
        self.img_cls = ctk.StringVar(value='NONE')
        self.gender = ctk.StringVar(value='NONE')
        self.age = ctk.StringVar(value='NONE')
        self.prompt = ctk.StringVar(value='')
        self.cfg_open = False
        self.cfg_files = []
        self.cfg_name = str(DEFAULT_JSON.relative_to(BASE_DIR)).replace('\\', '/')
        self.cfg_path = DEFAULT_JSON
        self.config = {}
        self.config_manager = ConfigManager(BASE_DIR, DEFAULT_JSON)
        self.loaded_loras = []
        self.recommended_ratio_sizes = RECOMMENDED_RATIO_SIZES
        self.show_orig = False
        self.load_env()
        self.model_var.set(DEFAULT_MODEL)
        self.protocol('WM_DELETE_WINDOW', self.close_app)
        self.ui()
        self.stdout_redirect, self.stderr_redirect = create_redirects(self.console)
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stderr_redirect
        self.after(100, lambda: self.state('zoomed'))
        self.load_start_cfg()
        if self.model_var.get():
            threading.Thread(target=self.load_models, args=(self.model_var.get(),), daemon=True).start()

    def close_app(self):
        if self.closing:
            return
        self.closing = True
        self.stop_requested.set()
        if self.save_job:
            try:
                self.after_cancel(self.save_job)
            except Exception:
                pass
        if hasattr(self, 'stdout_redirect'):
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        try:
            self.quit()
            self.destroy()
        finally:
            os._exit(0)

    def open_settings(self):
        from settings import open_settings_popup
        popup = open_settings_popup(self)
        popup.on_save = self.reload_after_settings

    def reload_after_settings(self):
        ConfigManager.load_env(BASE_DIR)
        configure_ctk_theme()
        self.after(50, self.restart_app)

    def restart_app(self):
        exe = str(Path(sys.executable).resolve())
        app_path = str(Path(__file__).resolve())
        subprocess.Popen([exe, app_path, *sys.argv[1:]], cwd=str(BASE_DIR))
        self.quit()
        self.destroy()

    def ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.left_panel = ctk.CTkFrame(self, fg_color='transparent')
        self.left_panel.grid(row=0, column=0, sticky='nsew', padx=(10, 5), pady=10)
        self.left_panel.grid_rowconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=0)
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_frame = ctk.CTkScrollableFrame(self.left_panel)
        self.left_frame.grid(row=0, column=0, sticky='nsew')
        self.left_frame.grid_columnconfigure(0, weight=0)
        self.left_frame.grid_columnconfigure(1, weight=1)
        self.input_label = ctk.CTkLabel(self.left_frame, text='INPUT IMAGE')
        self.input_label.grid(row=0, column=0, sticky='w', padx=4, pady=(0, 4))
        self.mask_button_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.mask_button_row.grid(row=0, column=1, sticky='e', padx=2, pady=(0, 4))
        self.reload_btn = GradientBtn(self.mask_button_row, text='↻', command=self.reload_mask, width=34, height=34, state='disabled')
        self.reload_btn.grid(row=0, column=0, sticky='e', padx=(0, 4))
        self.manual_btn = GradientBtn(self.mask_button_row, text='✎', command=self.toggle_manual_segment_mode, width=34, height=34, state='disabled')
        self.manual_btn.grid(row=0, column=1, sticky='e', padx=4)
        self.clear_seg_btn = GradientBtn(self.mask_button_row, text='🗑', command=self.clear_manual_segments, width=34, height=34, state='disabled')
        self.clear_seg_btn.grid(row=0, column=2, sticky='e', padx=(4, 0))
        self.img_box = ctk.CTkFrame(self.left_frame, fg_color='#030303', corner_radius=0, height=680)
        self.left_frame.grid_rowconfigure(1, minsize=680, weight=0)
        self.img_box.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=2, pady=(0, 8))
        self.img_box.grid_propagate(False)
        self.img_box.grid_rowconfigure(0, weight=1)
        self.img_box.grid_columnconfigure(0, weight=1)
        self.in_canvas = ctk.CTkCanvas(self.img_box, bg='#030303', highlightthickness=0)
        self.in_canvas.pack(fill='both', expand=True)
        self.ratio_var = ctk.StringVar(value='1:1')
        self.ratio_menu = ctk.CTkOptionMenu(self.img_box, variable=self.ratio_var, values=RATIO_OPTIONS, command=self.ratio_changed, width=104, height=34, corner_radius=6)
        self.ratio_menu.place(relx=1.0, x=-8, y=8, anchor='ne')
        self.reset_btn = GradientBtn(self.img_box, text='🖾', command=self.reset_crop, width=34, height=34)
        self.reset_btn.place(relx=1.0, x=-8, y=48, anchor='ne')
        self.upload_btn = GradientBtn(self.left_frame, text='UPLOAD IMAGE', command=self.upload, height=40)
        self.upload_btn.grid(row=2, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 6))
        self.prompt_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.prompt_row.grid(row=3, column=0, columnspan=2, sticky='ew', padx=2, pady=(7, 10))
        self.prompt_row.grid_columnconfigure(0, weight=0)
        self.prompt_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.prompt_row, text='ENTER YOUR PROMPT:', anchor='w', width=160).grid(row=0, column=0, sticky='w', padx=(2, 8))
        self.prompt_entry = ctk.CTkEntry(self.prompt_row, textvariable=self.prompt, height=38)
        self.prompt_entry.grid(row=0, column=1, sticky='ew', padx=(0, 2))
        self.prompt_entry.bind('<KeyRelease>', self.prompt_changed)
        self.cfg_box = ctk.CTkFrame(self.left_frame, corner_radius=8)
        self.cfg_box.grid_columnconfigure(0, weight=0)
        self.cfg_box.grid_columnconfigure(1, weight=1)
        self.cfg_box.grid_remove()
        self.preprocessing_row = ctk.CTkFrame(self.cfg_box, fg_color='transparent')
        self.preprocessing_row.grid(row=0, column=0, columnspan=2, sticky='ew', padx=2, pady=(6, 8))
        self.preprocessing_row.grid_columnconfigure(0, weight=1)
        self.preprocessing_row.grid_columnconfigure(1, weight=1)
        self.sd_rec_cb = ctk.CTkCheckBox(self.preprocessing_row, text='Recommended SD inpainting image', variable=self.sd_rec)
        self.sd_rec_cb.grid(row=0, column=0, sticky='w', padx=2, pady=3)
        self.apply_meta_cb = ctk.CTkCheckBox(self.preprocessing_row, text='Apply image class, age and gender', variable=self.apply_meta, command=self.toggle_meta_prompts)
        self.apply_meta_cb.grid(row=0, column=1, sticky='w', padx=2, pady=3)
        ctk.CTkLabel(self.cfg_box, text='MODEL:', anchor='w', width=100).grid(row=1, column=0, sticky='w', padx=(4, 8), pady=(0, 8))
        self.model_menu = ctk.CTkOptionMenu(self.cfg_box, variable=self.model_var, values=list(MODEL_OPTIONS.keys()), command=self.model_changed)
        self.model_menu.grid(row=1, column=1, sticky='ew', padx=2, pady=(0, 8))
        ctk.CTkLabel(self.cfg_box, text='LoRA:', anchor='w', width=100).grid(row=2, column=0, sticky='nw', padx=(4, 8), pady=(0, 8))
        self.lora_menu = GradientBtn(self.cfg_box, text='', command=self.toggle_lora_menu, anchor='w', height=34)
        self.lora_menu.grid(row=2, column=1, sticky='ew', padx=2, pady=(0, 4))
        self.lora_panel = ctk.CTkFrame(self.cfg_box, fg_color='transparent')
        self.lora_panel.grid(row=3, column=1, sticky='ew', padx=2, pady=(0, 8))
        self.lora_panel.grid_remove()
        self.refresh_lora_menu()
        self.meta_row = ctk.CTkFrame(self.cfg_box, fg_color='transparent')
        self.meta_row.grid(row=4, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
        self.meta_row.grid_columnconfigure(0, weight=0)
        self.meta_row.grid_columnconfigure(1, weight=1)
        self.meta_row.grid_columnconfigure(2, weight=0)
        self.meta_row.grid_columnconfigure(3, weight=1)
        self.meta_row.grid_columnconfigure(4, weight=0)
        self.meta_row.grid_columnconfigure(5, weight=1)
        ctk.CTkLabel(self.meta_row, text='IMAGE CLASS:', anchor='w', width=100).grid(row=0, column=0, sticky='w', padx=(2, 8))
        self.image_class_menu = ctk.CTkOptionMenu(self.meta_row, variable=self.img_cls, values=IMAGE_CLASS, command=self.prompt_sel_changed)
        self.image_class_menu.grid(row=0, column=1, sticky='ew', padx=(0, 8))
        ctk.CTkLabel(self.meta_row, text='GENDER:', anchor='w', width=75).grid(row=0, column=2, sticky='w', padx=(2, 8))
        self.gender_menu = ctk.CTkOptionMenu(self.meta_row, variable=self.gender, values=GENDER_CLASS, command=self.prompt_sel_changed)
        self.gender_menu.grid(row=0, column=3, sticky='ew', padx=(0, 8))
        ctk.CTkLabel(self.meta_row, text='AGE:', anchor='w', width=55).grid(row=0, column=4, sticky='w', padx=(2, 8))
        self.age_menu = ctk.CTkOptionMenu(self.meta_row, variable=self.age, values=['NONE', *AGE_CLASS], command=self.prompt_sel_changed)
        self.age_menu.grid(row=0, column=5, sticky='ew', padx=(0, 2))
        ctk.CTkLabel(self.cfg_box, text='JSON CONFIG:', anchor='w', width=100).grid(row=5, column=0, sticky='w', padx=(4, 8), pady=(0, 8))
        self.config_row = ctk.CTkFrame(self.cfg_box, fg_color='transparent')
        self.config_row.grid(row=5, column=1, sticky='ew', padx=2, pady=(0, 8))
        self.config_row.grid_columnconfigure(0, weight=1)
        self.config_row.grid_columnconfigure(1, weight=0)
        self.cfg_menu = ctk.CTkOptionMenu(self.config_row, values=[], command=self.config_changed)
        self.cfg_menu.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.autosave_btn = GradientBtn(self.config_row, text='AUTOSAVE: ON', command=self.toggle_autosave, height=38, width=105)
        self.autosave_btn.grid(row=0, column=1, sticky='e', padx=(5, 0))
        self.json = JSONTextBox(self.cfg_box, height=220, font_size=12, fg_color='#000000')
        self.json.grid(row=6, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
        self.json.set_change_callback(self.json_changed)
        self.out_opts = ctk.CTkFrame(self.cfg_box, fg_color='transparent')
        self.out_opts.grid(row=7, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 6))
        self.out_opts.grid_columnconfigure(0, weight=1)
        self.out_opts.grid_columnconfigure(1, weight=1)
        self.esrgan_cb = ctk.CTkCheckBox(self.out_opts, text='Enhance output image', variable=self.esrgan_out)
        self.esrgan_cb.grid(row=0, column=0, sticky='w', padx=2, pady=3)
        self.full_out_cb = ctk.CTkCheckBox(self.out_opts, text='Full image on output', variable=self.full_out)
        self.full_out_cb.grid(row=0, column=1, sticky='w', padx=2, pady=3)
        self.generate_container = ctk.CTkFrame(self.left_panel, corner_radius=8)
        self.generate_container.grid(row=1, column=0, sticky='ew', pady=(8, 0))
        self.generate_container.grid_columnconfigure(0, weight=1)
        self.generate_row = ctk.CTkFrame(self.generate_container, fg_color='transparent')
        self.generate_row.grid(row=0, column=0, sticky='ew', padx=8, pady=8)
        self.generate_row.grid_columnconfigure(0, weight=0)
        self.generate_row.grid_columnconfigure(1, weight=1)
        self.generate_row.grid_columnconfigure(2, weight=0)
        self.cfg_btn = GradientBtn(self.generate_row, text='CONFIG', command=self.toggle_config, width=70, height=42)
        self.cfg_btn.grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.generate_btn = GradientBtn(self.generate_row, text='GENERATE', command=self.generate, state='disabled', height=42)
        self.generate_btn.grid(row=0, column=1, sticky='ew', padx=5)
        self.chili_btn = GradientBtn(self.generate_row, text='🌶', command=self.chili_generate, state='disabled', width=42, height=42)
        self.chili_btn.grid(row=0, column=2, sticky='e', padx=(5, 0))
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 10), pady=10)
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.output_container = ctk.CTkFrame(self.right_frame, fg_color='#000000', corner_radius=6)
        self.output_container.grid(row=0, column=0, sticky='nsew', padx=10, pady=(10, 6))
        self.output_container.grid_rowconfigure(0, weight=1)
        self.output_container.grid_columnconfigure(0, weight=1)
        self.out_canvas = ctk.CTkCanvas(self.output_container, bg='#000000', highlightthickness=0)
        self.out_canvas.grid(row=0, column=0, sticky='nsew')
        self.swap_btn = GradientBtn(self.output_container, text='⇄', command=self.switch_output_image, width=34, height=34)
        self.swap_btn.place(relx=1.0, x=-8, y=8, anchor='ne')
        self.console = ConsoleTextBox(self.right_frame, height=260, wrap='none', font=('Consolas', 12), fg_color='#000000', text_color='#D0D0D0')
        self.console.grid(row=1, column=0, sticky='ew', padx=10, pady=6)
        self.save_clear_row = ctk.CTkFrame(self.right_frame, fg_color='transparent')
        self.save_clear_row.grid(row=2, column=0, sticky='ew', padx=10, pady=(6, 10))
        self.save_clear_row.grid_columnconfigure(0, weight=1)
        self.save_clear_row.grid_columnconfigure(1, weight=0)
        self.save_clear_row.grid_columnconfigure(2, weight=0)
        self.save_clear_row.grid_columnconfigure(3, weight=0)
        self.save_btn = GradientBtn(self.save_clear_row, text='SAVE IMAGE AS', command=self.save, state='disabled', height=40)
        self.save_btn.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.cf_btn = GradientBtn(self.save_clear_row, text='CF', command=self.save_compare, state='disabled', width=60, height=40)
        self.cf_btn.grid(row=0, column=1, sticky='e', padx=5)
        self.clear_btn = GradientBtn(self.save_clear_row, text='CLEAR', command=self.console.clear, height=40, width=100)
        self.clear_btn.grid(row=0, column=2, sticky='e', padx=(5, 0))
        self.settings_btn = GradientBtn(self.save_clear_row, text='⚙️', command=self.open_settings, height=40, width=46)
        self.settings_btn.grid(row=0, column=3, sticky='e', padx=(5, 0))
        self.in_canvas.bind('<Configure>', lambda e: self.show_input())
        self.in_canvas.bind('<ButtonPress-1>', self.start_crop)
        self.in_canvas.bind('<ButtonPress-3>', self.remove_manual_segment)
        self.in_canvas.bind('<B1-Motion>', self.update_crop_selection)
        self.in_canvas.bind('<ButtonRelease-1>', self.finish_crop)
        self.out_canvas.bind('<Configure>', lambda e: self.show_output())
        self.update_autosave()
        self.show_input()
        self.show_output()
        self.after(80, self.spin_loader)

    @staticmethod
    def normalize_choice(value, valid, fallback='NONE'):
        value = str(value).strip()
        if not value or value.upper() == 'NONE':
            return fallback
        value = value.lower()
        return value if value in valid else fallback

    def set_widget_state(self, state, *widgets):
        for widget in widgets:
            if widget is not None:
                widget.configure(state=state)

    def schedule(self, callback, *args, **kwargs):
        self.after(0, lambda: callback(*args, **kwargs))

    def input_is_busy(self):
        return any((self.processing, self.model_loading, self.seg_loading, self.class_loading, self.manual_loading))

    def spin_loader(self):
        busy = self.input_is_busy()
        if busy != self.input_busy_visual:
            self.input_busy_visual = busy
            self.show_input()
        self.after(80, self.spin_loader)

    def toggle_config(self):
        self.cfg_open = not self.cfg_open
        if self.cfg_open:
            self.cfg_box.grid(row=4, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
            self.cfg_btn.configure(fg_color='#3c8dbc', hover_color='#3279a3')
            self.after(50, lambda: self.left_frame._parent_canvas.yview_moveto(1.0))
        else:
            self.cfg_box.grid_remove()
            self.cfg_btn.configure(fg_color=ctk.ThemeManager.theme['CTkButton']['fg_color'], hover_color=ctk.ThemeManager.theme['CTkButton']['hover_color'])

    def refresh_lora_menu(self):
        if not hasattr(self, 'lora_panel'):
            return
        for widget in self.lora_panel.winfo_children():
            widget.destroy()
        self.lora_vars = {}
        for row, path in enumerate(self.lora_paths):
            variable = ctk.BooleanVar(value=path in self.selected_loras)
            self.lora_vars[path] = variable
            ctk.CTkCheckBox(self.lora_panel, text=Path(path).stem, variable=variable, command=lambda path=path, variable=variable: self.lora_changed(path, variable)).grid(row=row, column=0, sticky='w', padx=4, pady=2)
        selected = sum(variable.get() for variable in self.lora_vars.values())
        self.lora_menu.configure(text=f'{selected} LoRA selected [Select]')

    def toggle_lora_menu(self):
        if self.lora_panel.winfo_ismapped():
            self.lora_panel.grid_remove()
        else:
            self.lora_panel.grid()

    def lora_changed(self, path, variable):
        if variable.get():
            self.selected_loras.add(path)
        else:
            self.selected_loras.discard(path)
        self.apply_lora_selection()
        self.refresh_lora_menu()

    def apply_lora_selection(self):
        if self.pipe is None or not self.loaded_loras:
            return
        names = [name for path, name in self.loaded_loras if path in self.selected_loras]
        if names:
            self.pipe.set_adapters(names)
        else:
            self.pipe.disable_lora()

    def update_gen_state(self):
        if self.processing:
            self.generate_btn.configure(state='normal', text='STOP GENERATING', command=self.stop_generation)
            self.chili_btn.configure(state='disabled')
            return
        state = 'normal' if self.models_ready and self.original_image is not None and not self.model_loading and not self.seg_loading and not self.class_loading and not self.manual_loading else 'disabled'
        self.generate_btn.configure(state=state, text='GENERATE', command=self.generate)
        self.chili_btn.configure(state=state)

    def chili_generate(self):
        if self.processing:
            return
        try:
            payload = open_payload(CHILI_BIN)
            self.generate(config_override=dict(payload))
        except Exception:
            print('Nyaaahhh~ I don\'t want the chili! It\'s too spicy for me ⸜(｡˃ ᵕ ˂ )⸝♡')
        return

    def stop_generation(self):
        if not self.processing:
            return
        self.stop_requested.set()
        print('Stopping...')
        self.generate_btn.configure(state='disabled', text='STOPPING...')

    def check_stop_requested(self):
        if self.stop_requested.is_set():
            raise GenerationStopped()

    def reset_preview_state(self):
        self.mask_image = None
        self.mask_src = None
        self.sd_input_image = None
        self.output_image = None
        self.show_orig = False

    def refresh_crop_ui(self, *, allow_reload=True):
        self.reload_btn.configure(state='normal' if self.original_image is not None and not self.processing and allow_reload else 'disabled')
        self.update_manual_btns()
        self.update_crop_state()
        self.update_gen_state()

    def update_crop_state(self):
        state = 'normal' if self.original_image is not None and not self.processing and not self.seg_loading and not self.manual_loading else 'disabled'
        if self.manual_mode:
            state = 'disabled'
        self.reset_btn.configure(state=state)
        self.ratio_menu.configure(state=state)
        self.update_manual_btns()

    def load_env(self):
        global MODEL_OPTIONS, DEFAULT_MODEL
        MODEL_OPTIONS, DEFAULT_MODEL, self.lora_paths, self.start_cfg, autosave = self.config_manager.read_runtime_settings(MODEL_DIR)
        self.selected_loras = set(self.lora_paths)
        if not MODEL_OPTIONS:
            print(f'Missing inpainting models: no .safetensors files found in {MODEL_DIR} or configured SD_INPAINT_MODEL paths.')
        self.autosave.set(autosave)

    def write_env(self):
        self.config_manager.write_env_settings(self.cfg_path, self.autosave.get())

    def rel_cfg(self):
        return self.config_manager.relative_config_path(self.cfg_path)

    def rel_disp(self, path):
        return self.config_manager.relative_display_path(path)

    def refresh_cfgs(self):
        self.cfg_files = self.config_manager.refresh_config_files(BASE_DIR)
        values = [self.rel_disp(p) for p in self.cfg_files]
        fallback = self.rel_disp(DEFAULT_JSON)
        self.cfg_menu.configure(values=values if values else [fallback])

    def load_start_cfg(self):
        self.refresh_cfgs()
        candidates = self.cfg_files.copy()
        target = self.start_cfg
        if not target.exists() or target.suffix.lower() != '.json':
            target = DEFAULT_JSON if DEFAULT_JSON.exists() else candidates[0] if candidates else target
        if target not in candidates and target.exists():
            candidates.append(target)
            candidates = sorted(candidates, key=lambda p: p.name.lower())
            self.cfg_files = candidates
            self.cfg_menu.configure(values=[self.rel_disp(p) for p in candidates])
        if target.exists():
            self.load_config(target)
            self.cfg_menu.set(self.rel_disp(target))
        self.update_autosave()
        self.write_env()

    def config_changed(self, choice):
        if self.processing or self.model_loading or self.seg_loading or self.class_loading:
            return
        target = BASE_DIR / choice
        if not target.exists():
            return
        try:
            self.load_config(target)
            self.write_env()
        except Exception as e:
            print(f'Configuration Error: {e}')

    def toggle_autosave(self):
        if self.processing:
            return
        self.autosave.set(not self.autosave.get())
        self.update_autosave()
        self.write_env()
        if self.autosave.get():
            self.save_json()

    def update_autosave(self):
        if not hasattr(self, 'autosave_btn'):
            return
        if self.autosave.get():
            self.autosave_btn.configure(text='AUTOSAVE: ON', fg_color='#1f8f3a', hover_color='#176b2c')
        else:
            self.autosave_btn.configure(text='AUTOSAVE: OFF', fg_color='#666666', hover_color='#555555')

    def cleanup_gpu(self):
        try:
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
        except Exception:
            pass
        gc.collect()
        if DEVICE == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def load_config(self, path):
        self.config, self.cfg_path = self.config_manager.load_config(path)
        self.cfg_name = self.rel_cfg()
        self.json.delete('1.0', 'end')
        self.json.insert('1.0', json.dumps(self.config, indent=4, ensure_ascii=False))
        self.prompt.set(str(self.config.get('positive_prompt', '')))
        if hasattr(self, 'config_menu'):
            self.cfg_menu.set(self.rel_disp(self.cfg_path))
        print(f'Loaded {Path(self.cfg_name).name}')

    def prompt_changed(self, event=None):
        text = self.json.get('1.0', 'end').strip()
        if not text:
            return
        try:
            data = json.loads(text)
        except Exception:
            return
        data['positive_prompt'] = self.prompt.get()
        self.config = data
        updated_text = json.dumps(data, indent=4, ensure_ascii=False)
        if text == updated_text:
            return
        self.json.delete('1.0', 'end')
        self.json.insert('1.0', updated_text)

    def json_changed(self, event=None):
        text = self.json.get('1.0', 'end').strip()
        if not text:
            return
        try:
            self.config = self.config_manager.sync_config(self.cfg_path, text, False)
        except Exception as e:
            print(f'JSON error: {e}')
            return
        if self.save_job:
            try:
                self.after_cancel(self.save_job)
            except Exception:
                pass
        if self.autosave.get():
            self.save_job = self.after(700, self.save_json)

    def save_json(self):
        self.save_job = None
        if not self.autosave.get():
            return
        try:
            text = self.json.get('1.0', 'end').strip()
            self.config = self.config_manager.save_json(self.cfg_path, text, self.autosave.get())
            self.write_env()
            print('Config saved')
        except Exception as e:
            print(f'JSON error: {e}')

    def sync_config(self):
        text = self.json.get('1.0', 'end').strip()
        self.config = self.config_manager.sync_config(self.cfg_path, text, self.autosave.get())

    def unload_pipe(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.model_name = None
        self.dtype = None
        self.cleanup_gpu()

    def restore_diffusion_model(self):
        if self.pipe is None:
            return
        print(f'Model → {DEVICE.upper()}')
        if DEVICE == 'cuda' and self.dtype == torch.float16:
            self.pipe.to(DEVICE, dtype=torch.float16)
        else:
            self.pipe.to(DEVICE)
        self.cleanup_gpu()

    def model_changed(self, choice):
        if self.processing or self.model_loading:
            return
        if choice not in MODEL_OPTIONS:
            return
        if self.model_name == choice and self.models_ready:
            return
        self.config_manager.reorder_model_list_in_env(MODEL_OPTIONS.get(choice, ''))
        self.models_ready = False
        self.generate_btn.configure(state='disabled')
        self.model_menu.configure(state='disabled')
        self.model_loading = True
        threading.Thread(target=self.load_models, args=(choice,), daemon=True).start()

    def load_models(self, model_name):
        try:
            model_id = MODEL_OPTIONS.get(model_name, '')
            if not model_id or not Path(model_id).exists():
                raise FileNotFoundError(f'Model "{model_name}" was not found. Add a .safetensors file to the model folder or list it in "SD_INPAINT_MODEL" in .env.')
            self.unload_pipe()
            dtype = torch.float16 if DEVICE == 'cuda' else torch.float32
            print(f'Loading {model_name}')
            pipe = StableDiffusionInpaintPipeline.from_single_file(model_id, torch_dtype=dtype, safety_checker=None, local_files_only=True)
            lora_paths = self.config_manager.discover_loras()
            loaded_loras = []
            for index, lora_path in enumerate(lora_paths):
                lora_name = f'lora_{index}'
                try:
                    pipe.load_lora_weights(lora_path, adapter_name=lora_name)
                    loaded_loras.append((lora_path, lora_name))
                    print(f'LoRA loaded: {lora_path}')
                except Exception as error:
                    print(f'LoRA not loaded: {lora_path} ({error})')
            names = [name for path, name in loaded_loras if path in self.selected_loras]
            if names:
                pipe.set_adapters(names)
            elif loaded_loras:
                pipe.disable_lora()
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(DEVICE)
            pipe.set_progress_bar_config(disable=True)
            pipe.enable_attention_slicing()
            if DEVICE == 'cuda':
                try:
                    pipe.enable_vae_slicing()
                except Exception:
                    pass
            self.pipe = pipe
            self.lora_paths = lora_paths
            self.loaded_loras = loaded_loras
            self.model_name = model_name
            self.dtype = dtype
            self.cleanup_gpu()
            self.models_ready = True
            self.model_loading = False
            print(f'{model_name} ready • {DEVICE.upper()}')
            self.schedule(self.refresh_lora_menu)
            self.schedule(self.model_menu.configure, state='normal')
            self.schedule(self.update_gen_state)
        except Exception as e:
            self.models_ready = False
            self.model_loading = False
            self.seg_loading = False
            print(f'Error: {e}')
            self.cleanup_gpu()
            self.schedule(self.model_menu.configure, state='normal')
            self.schedule(print, f'Model Error: {e}')

    def set_img_class(self, classification):
        best_class = self.normalize_choice(classification.get('best_class', ''), IMAGE_CLASS)
        if best_class not in IMAGE_CLASS:
            best_class = IMAGE_CLASS[1]
        self.img_cls.set(best_class)
        self.image_class_menu.configure(state='normal')

    def set_gender(self, result):
        detected_gender = self.normalize_choice(result[0], GENDER_CLASS)
        self.gender.set(detected_gender)
        self.gender_menu.configure(state='normal')

    def set_age(self, result):
        detected_age = self.normalize_choice(result[0], ['NONE', *AGE_CLASS])
        self.age.set(detected_age)
        self.age_menu.configure(state='normal')

    def prep_image(self, path):
        image = ImageOps.exif_transpose(load_image(path))
        has_alpha = image.mode in ('RGBA', 'LA') or 'transparency' in image.info
        if has_alpha:
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert('RGB')
            print('Alpha background filled')
        else:
            image = image.convert('RGB')
        return image

    def classify_upload(self, file_path):
        try:
            print('Classifying image...')
            classification = self.classify_input_image(file_path)
            self.cls_res = classification
            self.schedule(self.set_img_class, classification)
            gender_result = predict_gender(file_path)
            self.gender_res = gender_result
            self.schedule(self.set_gender, gender_result)
            age_result = predict_age(file_path)
            self.cleanup_gpu()
            self.age_res = age_result
            self.schedule(self.set_age, age_result)
            best_class = self.normalize_choice(classification.get('best_class', ''), IMAGE_CLASS)
            best_class = best_class if best_class in IMAGE_CLASS else IMAGE_CLASS[1]
            detected_gender = self.normalize_choice(gender_result[0], GENDER_CLASS)
            detected_age = self.normalize_choice(age_result[0], ['NONE', *AGE_CLASS])
            print(f'Class: {best_class.upper()}, Gender: {detected_gender.upper()}, Age: {detected_age.upper()}')
            print('Classification finished')
            self.schedule(self.set_widget_state, 'normal', self.image_class_menu, self.gender_menu, self.age_menu)
        except Exception as e:
            self.cls_res = None
            self.gender_res = None
            self.age_res = None
            self.schedule(print, f'Image Classification Error: {e}')
            self.schedule(self.set_widget_state, 'normal', self.image_class_menu, self.gender_menu, self.age_menu)
            self.schedule(self.img_cls.set, 'NONE')
            self.schedule(self.gender.set, 'NONE')
            self.schedule(self.age.set, 'NONE')
            print(f'Classify error: {e}')
        finally:
            self.class_loading = False
            self.schedule(self.update_gen_state)
            self.schedule(self.update_crop_state)

    def upload(self):
        if self.processing or self.model_loading or self.seg_loading or self.class_loading or self.manual_mode or self.manual_loading:
            return
        path = filedialog.askopenfilename(title='Select input image', filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.bmp *.avif')])
        if not path:
            return
        try:
            image = self.prep_image(path)
            self.input_path = path
            self.img_name = Path(path).name
            self.crop_box = None
            self.cropping = False
            self.crop_handle = None
            self.crop_start = None
            self.original_image = image.copy()
            self.ratio_var.set('FREE')
            self.crop_box = self.crop_to_ratio('FREE')
            self.input_image = image.copy()
            self.sd_input_image = None
            self.output_image = None
            self.show_orig = False
            self.mask_image = None
            self.mask_src = None
            self.cls_res = None
            self.gender_res = None
            self.age_res = None
            self.class_loading = True
            self.img_cls.set('NONE')
            self.gender.set('NONE')
            self.age.set('NONE')
            self.image_class_menu.configure(state='disabled')
            self.gender_menu.configure(state='disabled')
            self.age_menu.configure(state='disabled')
            self.save_btn.configure(state='disabled')
            self.cf_btn.configure(state='disabled')
            self.generate_btn.configure(state='disabled')
            self.reload_btn.configure(state='normal')
            self.manual_btn.configure(state='normal')
            self.clear_seg_btn.configure(state='normal')
            self.reset_btn.configure(state='normal')
            self.show_input()
            self.show_output()
            print(f'Loaded {self.img_name}')
            threading.Thread(target=self.classify_upload, args=(path,), daemon=True).start()
        except Exception as e:
            print(f'Image Error: {e}')

    def build_prompts(self, config, selected_class, selected_gender, selected_age, apply_class_gender):
        positive_prompt = str(config.get('positive_prompt', '')).strip()
        negative_prompt = str(config.get('negative_prompt', '')).strip()
        if not apply_class_gender:
            return positive_prompt, negative_prompt
        class_prompts = {'NONE': ('', '')}
        for class_name in IMAGE_CLASS[1:]:
            class_prompts[class_name] = (class_name, ', '.join(other for other in IMAGE_CLASS[1:] if other != class_name))
        selected_class = self.normalize_choice(selected_class, IMAGE_CLASS)
        selected_gender = self.normalize_choice(selected_gender, GENDER_CLASS)
        selected_age = self.normalize_choice(selected_age, ['NONE', *AGE_CLASS])
        gender_append = '' if selected_gender == 'NONE' else selected_gender
        age_append = '' if selected_age == 'NONE' else selected_age
        class_positive, class_negative = class_prompts.get(selected_class, ('', ''))
        person_prefix = ' '.join(part for part in (gender_append, age_append) if part)
        if person_prefix:
            positive_prompt = f'{person_prefix}, {positive_prompt}' if positive_prompt else person_prefix
        if class_positive:
            positive_prompt = f'{class_positive} {positive_prompt}' if positive_prompt else class_positive
        if class_negative:
            negative_prompt = f'{class_negative}, {negative_prompt}' if negative_prompt else class_negative
        return positive_prompt, negative_prompt

    def log_actual_prompts(self):
        positive_prompt, negative_prompt = self.build_prompts(self.config, self.img_cls.get(), self.gender.get(), self.age.get(), True)
        print(f'Positive Prompt: {positive_prompt}')
        print(f'Negative Prompt: {negative_prompt}')

    def prompt_sel_changed(self, choice):
        if self.processing or not self.apply_meta.get():
            return
        self.log_actual_prompts()

    def toggle_meta_prompts(self):
        if self.apply_meta.get():
            self.log_actual_prompts()

    def generate(self, config_override=None):
        if self.processing:
            return
        if self.model_loading or self.seg_loading:
            print('Models loading')
            return
        if self.class_loading:
            print('Classification running')
            return
        if not self.models_ready:
            print('Model not ready')
            return
        if self.manual_mode:
            self.disable_manual_segment_mode()
        if self.original_image is None:
            print('No image')
            return
        if config_override is None:
            try:
                self.sync_config()
            except Exception as e:
                print(f'Configuration Error: {e}')
                return
        self.gen_count += 1
        generation_id = self.gen_count
        source = self.sd_input_image.copy() if self.sd_input_image is not None else None
        mask = self.mask_image.copy() if self.has_valid_mask(self.mask_image) else None
        if mask is None:
            self.mask_src = None
        elif self.mask_src == 'manual':
            print('Manual segment available • skipping automatic segmentation')
        elif self.mask_src == 'auto':
            print('Automatic segment available • reusing existing mask')
        else:
            print('Existing mask available • skipping automatic segmentation')
        original_image = self.original_image.copy()
        crop_box = self.get_effective_crop_box()
        config = dict(config_override) if config_override is not None else dict(self.config)
        model_name = str(self.model_name or self.model_var.get())
        selected_class = self.img_cls.get()
        selected_gender = self.gender.get()
        selected_age = self.age.get()
        apply_class_gender = bool(self.apply_meta.get())
        full_image_output = bool(self.full_out.get())
        self.stop_requested.clear()
        esrgan_output = bool(self.esrgan_out.get())
        autosave = bool(self.autosave.get())
        self.processing = True
        self.output_image = None
        self.show_orig = False
        self.generate_btn.configure(state='disabled')
        self.upload_btn.configure(state='disabled')
        self.reload_btn.configure(state='disabled')
        self.manual_btn.configure(state='disabled')
        self.save_btn.configure(state='disabled')
        self.cf_btn.configure(state='disabled')
        self.model_menu.configure(state='disabled')
        self.image_class_menu.configure(state='disabled')
        self.gender_menu.configure(state='disabled')
        self.age_menu.configure(state='disabled')
        self.ratio_menu.configure(state='disabled')
        self.start_time = time.time()
        self.show_output()
        print(f'Generation {generation_id} start')
        self.after(0, self.update_gen_state)
        threading.Thread(target=self.worker, args=(source, mask, original_image, crop_box, config, model_name, selected_class, selected_gender, selected_age, apply_class_gender, full_image_output, esrgan_output, autosave, generation_id), daemon=True).start()

    def classify_input_image(self, file_path):
        return classify_image(file_path)

    def composite_mask(self, base, generated, mask):
        if generated.size != base.size:
            generated = generated.resize(base.size, Image.LANCZOS)
        if mask.size != base.size:
            mask = mask.resize(base.size, Image.Resampling.NEAREST)
        return Image.composite(generated, base, mask).convert('RGB')

    def worker(self, source, mask, original_image, crop_box, config, model_name, selected_class, selected_gender, selected_age, apply_class_gender, full_image_output, esrgan_output, autosave, generation_id):
        try:
            steps = int(config.get('steps'))
            cfg = float(config.get('cfg'))
            strength = float(config.get('strength'))
            seed = int(config.get('seed', -1))
            cfg_rescale = float(config.get('cfg_rescale'))
            positive_prompt, negative_prompt = self.build_prompts(config, selected_class, selected_gender, selected_age, apply_class_gender)
            if source is None:
                print('Preprocessing...')
                source = self.prep_input(original_image)
            else:
                source = source.copy()
            if not self.has_valid_mask(mask):
                self.seg_loading = True
                self.after(0, self.update_crop_state)
                print('No segment is present on the input preview • running automatic segmentation...')
                mask = self.make_mask(source, config)
                self.mask_image = mask.copy()
                self.mask_src = 'auto'
                self.after(0, self.show_input)
                self.seg_loading = False
                self.after(0, self.update_crop_state)
            elif mask.size != source.size:
                print(f'Reusing {self.mask_src or "existing"} segment • resizing mask to processed input')
                mask = mask.resize(source.size, Image.Resampling.NEAREST)
                self.mask_image = mask.copy()
            if np.asarray(mask, dtype=np.uint8).max() < 10:
                raise RuntimeError('No selected segmentation area was detected by segdinosam2.')
            self.check_stop_requested()
            w, h = self.generation_size(*source.size)
            init = source.resize((w, h), Image.LANCZOS)
            mask = mask.resize((w, h), Image.Resampling.NEAREST)
            self.check_stop_requested()
            if seed == -1:
                seed = torch.randint(0, 2 ** 32 - 1, (1,), device='cpu').item()
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            if self.pipe is None:
                raise RuntimeError('Selected diffusion model is not loaded.')
            print(f'Seed {seed}')
            self.restore_diffusion_model()
            scheduler_config = dict(self.pipe.scheduler.config)

            def progress(pipe, step_index, timestep, callback_kwargs):
                self.check_stop_requested()
                step = step_index + 1
                elapsed = time.time() - self.start_time
                remaining = max(0, (steps - step) * (elapsed / max(step, 1)))
                percent = int(step / steps * 100)
                bar_len = 30
                filled = round(percent / 100 * bar_len)
                bar = f'{"█" * filled}{"░" * (bar_len - filled)}'
                self.console.log(f'[{bar}] {percent:3d}% | {step}/{steps} | [{self.time_text(elapsed)}<{self.time_text(remaining)}]', live=True)
                return callback_kwargs

            print(f'Generating {w}x{h}...')
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
            with torch.inference_mode():
                result = self.pipe(prompt=positive_prompt, negative_prompt=negative_prompt, image=init, mask_image=mask, num_inference_steps=steps, guidance_scale=cfg, strength=strength, generator=generator, width=w, height=h, callback_on_step_end=progress, guidance_rescale=cfg_rescale)
            self.check_stop_requested()
            generated = result.images[0].convert('RGB')
            del result
            generated = self.composite_mask(init, generated, mask)
            if esrgan_output:
                generated = enhance(generated, output=True, logger=self.console.log, output_target=OUTPUT_TARGET)
            self.check_stop_requested()
            if full_image_output:
                final_image = self.overlay_generated_crop(original_image, generated, crop_box)
            else:
                final_image = self.resize_output(generated)
            elapsed = time.time() - self.start_time

            def finish_generation():
                self.output_image = final_image
                self.show_orig = False
                self.show_output()
                self.save_btn.configure(state='normal')
                self.cf_btn.configure(state='normal')
                if autosave:
                    self.save()

            self.after(0, finish_generation)
            print(f'\nGeneration {generation_id} done • {self.time_text(elapsed)}')
        except GenerationStopped:
            print(f'Generation {generation_id} stopped')
        except Exception as e:
            print(f'Generation {generation_id} error: {e}')
            self.schedule(print, f'Generation Error: {e}')
        finally:
            self.processing = False
            self.stop_requested.clear()
            self.seg_loading = False
            self.cleanup_gpu()
            self.schedule(self.update_gen_state)
            self.schedule(self.reload_btn.configure, state='normal' if self.original_image is not None and not self.seg_loading else 'disabled')
            self.schedule(self.update_crop_state)
            self.schedule(self.upload_btn.configure, state='normal')
            self.schedule(self.model_menu.configure, state='normal')
            self.schedule(self.image_class_menu.configure, state='normal')
            self.schedule(self.gender_menu.configure, state='normal')
            self.schedule(self.age_menu.configure, state='normal')
            self.schedule(self.ratio_menu.configure, state='normal')

    def time_text(self, seconds):
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        return f'{minutes:02d}:{seconds:02d}'

    def show_input(self):
        if not hasattr(self, 'in_canvas'):
            return
        self.in_canvas.delete('all')
        image = self.input_image if self.input_image is not None else self.original_image
        if image is None:
            self.input_box = None
            canvas_w = max(1, self.in_canvas.winfo_width())
            canvas_h = max(1, self.in_canvas.winfo_height())
            self.in_canvas.create_text(canvas_w // 2, canvas_h // 2, text='Upload Image', fill='#888888', font=('Segoe UI', 22))
            self.update_crop_state()
            return
        canvas_w = max(1, self.in_canvas.winfo_width())
        canvas_h = max(1, self.in_canvas.winfo_height())
        fitted = ImageOps.contain(image, (canvas_w, canvas_h), method=Image.Resampling.LANCZOS)
        x = (canvas_w - fitted.width) // 2
        y = (canvas_h - fitted.height) // 2
        self.input_box = (x, y, fitted.width, fitted.height)
        if self.mask_image is not None:
            display_mask = self.compose_mask_for_display(image, fitted.size)
            overlay = Image.new('RGBA', fitted.size, (0, 0, 0, 0))
            red = Image.new('RGBA', fitted.size, (255, 0, 0, 90))
            overlay = Image.composite(red, overlay, display_mask)
            fitted = Image.alpha_composite(fitted.convert('RGBA'), overlay).convert('RGB')
        if self.input_is_busy():
            fitted = ImageEnhance.Brightness(fitted).enhance(0.45)
        self.input_pic = ImageTk.PhotoImage(fitted)
        self.in_canvas.create_image(x, y, anchor='nw', image=self.input_pic)
        self.draw_crop_overlay()
        if self.input_is_busy():
            self.play_gif(LOADING_GIF, x=(x + fitted.width / 2), y=(y + fitted.height / 2), anchor='center', speed=1.2)
        else:
            self.stop_gif(LOADING_GIF)
        self.update_crop_state()

    def show_output(self):
        if not hasattr(self, 'out_canvas'):
            return
        self.out_canvas.delete('all')
        comparison_image = self.original_image if self.full_out.get() else self.sd_input_image
        image = comparison_image if self.show_orig else self.output_image
        if image is None:
            canvas_w = max(1, self.out_canvas.winfo_width())
            canvas_h = max(1, self.out_canvas.winfo_height())
            if self.processing and self.original_image is not None:
                fitted = ImageOps.contain(self.original_image, (canvas_w, canvas_h), method=Image.Resampling.LANCZOS)
                fitted = ImageEnhance.Brightness(fitted).enhance(0.25).filter(ImageFilter.GaussianBlur(6))
                self.output_pic = ImageTk.PhotoImage(fitted)
                x = (canvas_w - fitted.width) // 2
                y = (canvas_h - fitted.height) // 2
                self.out_canvas.create_image(x, y, anchor='nw', image=self.output_pic)
                self.play_gif(LOADING_GIF, self.out_canvas, canvas_w / 2, canvas_h / 2, speed=1.2)
            else:
                self.stop_gif(LOADING_GIF, self.out_canvas)
                label = 'Waiting to generate...' if self.original_image is not None else 'Output Image'
                self.out_canvas.create_text(canvas_w // 2, canvas_h // 2, text=label, fill='#888888', font=('Segoe UI', 18))
            return
        self.stop_gif(LOADING_GIF, self.out_canvas)
        canvas_w = max(1, self.out_canvas.winfo_width())
        canvas_h = max(1, self.out_canvas.winfo_height())
        fitted = ImageOps.contain(image, (canvas_w, canvas_h), method=Image.Resampling.LANCZOS)
        self.output_pic = ImageTk.PhotoImage(fitted)
        x = (canvas_w - fitted.width) // 2
        y = (canvas_h - fitted.height) // 2
        self.out_canvas.create_image(x, y, anchor='nw', image=self.output_pic)

    def switch_output_image(self):
        if self.output_image is None and self.original_image is None and self.sd_input_image is None:
            return
        self.show_orig = not self.show_orig
        self.show_output()

    def save(self):
        if self.output_image is None:
            return
        path = prompt_save_path('Save generated image', f'{Path(self.img_name or "image").stem}_DiffuVision', self.input_path, self.last_dir)
        if path is None:
            return
        self.last_dir = path.parent
        try:
            output = self.output_image
            if path.suffix.lower() in ('.jpg', '.jpeg'):
                output = output.convert('RGB')
            output.save(path)
            print('Saved')
        except Exception as e:
            print(f'Save Error: {e}')

    def save_compare(self):
        if self.original_image is None or self.output_image is None:
            return
        path = prompt_save_path('Save comparison image', f'{Path(self.img_name or "image").stem}_DiffuVision_CF', self.input_path, self.last_dir)
        if path is None:
            return
        self.last_dir = path.parent
        try:
            original = self.original_image.convert('RGB')
            output = self.output_image.convert('RGB')
            target_height = max(original.height, output.height)
            left_width = int(original.width * target_height / original.height)
            right_width = int(output.width * target_height / output.height)
            original = original.resize((left_width, target_height), Image.Resampling.LANCZOS)
            output = output.resize((right_width, target_height), Image.Resampling.LANCZOS)
            comparison = Image.new('RGB', (left_width + right_width, target_height), (0, 0, 0))
            comparison.paste(original, (0, 0))
            comparison.paste(output, (left_width, 0))
            if path.suffix.lower() in ('.jpg', '.jpeg'):
                comparison = comparison.convert('RGB')
            comparison.save(path)
            print('Comparison Saved')
        except Exception as e:
            print(f'Comparison Save Error: {e}')

if __name__ == '__main__':
    app = App()
    app.mainloop()