import os, sys, json, time, threading, gc
from pathlib import Path
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageEnhance, ImageTk, ImageOps
import numpy as np, torch
import torchvision.transforms.functional as TF
sys.modules.setdefault('torchvision.transforms.functional_tensor', TF)
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from modules.img_classify import AGE_CLASS, GENDER_CLASS, IMAGE_CLASS, classify_image, predict_age, predict_gender
from widgets.json_textbox import JSONTextBox
from widgets.console_textbox import ConsoleTextBox, create_redirects
from utils.config_manager import ConfigManager
from utils.access_gate import open_payload
from utils.image_loader import load_image
from modules.upscale_img import enhance
from modules.segment_img import SegmentImageMixin
from modules.sd_ideal import SDIdealImageMixin, OUTPUT_TARGET, RECOMMENDED_RATIO_SIZES
from modules.crop_img import CropImageMixin

BASE_DIR = Path(__file__).resolve().parent
ctk.set_appearance_mode('system')
ctk.set_default_color_theme(str(BASE_DIR / 'themes' / 'red.json'))
MODEL_DIR = BASE_DIR / 'model'
DEFAULT_JSON = BASE_DIR / 'config/sd/default.json'
CHILI_BIN = BASE_DIR / 'config/sd/chili.bin'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_OPTIONS = {}
DEFAULT_MODEL = ''
RATIO_OPTIONS = ['FREE', '1:1', '4:3', '3:2', '16:9', '5:4', '4:5', '3:4', '2:3', '9:16']
class GenerationStopped(Exception):
    pass

class App(SegmentImageMixin, CropImageMixin, SDIdealImageMixin, ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('DiffuVision - Inpainting with Stable Diffusion')
        self.iconbitmap('favicon.ico')
        self.geometry('1600x1150')
        self.minsize(1100, 900)
        self.pipe = None
        self.current_model_name = None
        self.pipeline_dtype = None
        self.segmentation = None
        self.manual_segmenter = None
        self.manual_selection_image = None
        self.manual_segment_mode = False
        self.manual_segment_loading = False
        self.original_image = None
        self.input_image = None
        self.sd_input_image = None
        self.output_image = None
        self.input_path = None
        self.input_image_name = None
        self.last_save_dir = None
        self.generation_counter = 0
        self.mask_image = None
        self.mask_source = None
        self.classification_result = None
        self.gender_result = None
        self.age_result = None
        self.classification_loading = False
        self.processing = False
        self.models_ready = False
        self.model_loading = False
        self.segmentation_loading = False
        self.start_time = 0
        self.save_job = None
        self.input_photo = None
        self.output_photo = None
        self.input_loading_visual = False
        self.input_spinner_id = None
        self.input_spinner_angle = 0
        self.crop_box = None
        self.crop_box_start = None
        self.input_display_info = None
        self.cropping = False
        self.active_crop_handle = None
        self.recommended_sd_var = ctk.BooleanVar(value=True)
        self.esrgan_output_var = ctk.BooleanVar(value=False)
        self.full_image_output_var = ctk.BooleanVar(value=True)
        self.apply_class_gender_var = ctk.BooleanVar(value=False)
        self.stop_requested = threading.Event()
        self.closing = False
        self.autosave_var = ctk.BooleanVar(value=True)
        self.model_var = ctk.StringVar(value='')
        self.image_class_var = ctk.StringVar(value='NONE')
        self.gender_var = ctk.StringVar(value='NONE')
        self.age_var = ctk.StringVar(value='NONE')
        self.positive_prompt_var = ctk.StringVar(value='')
        self.config_visible = False
        self.config_files = []
        self.active_config_name = str(DEFAULT_JSON.relative_to(BASE_DIR)).replace('\\', '/')
        self.active_config_path = DEFAULT_JSON
        self.config = {}
        self.config_manager = ConfigManager(BASE_DIR, DEFAULT_JSON)
        self.recommended_ratio_sizes = RECOMMENDED_RATIO_SIZES
        self.output_showing_original = False
        self.load_env_settings()
        self.model_var.set(DEFAULT_MODEL)
        self.protocol('WM_DELETE_WINDOW', self.close_app)
        self.ui()
        self.stdout_redirect, self.stderr_redirect = create_redirects(self.console)
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stderr_redirect
        self.after(100, self.maximize)
        self.load_startup_config()
        if self.model_var.get():
            threading.Thread(target=self.load_models, args=(self.model_var.get(),), daemon=True).start()

    def maximize(self):
        try:
            self.state('zoomed')
        except Exception:
            pass

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
        self.reload_mask_btn = ctk.CTkButton(self.mask_button_row, text='↻', command=self.reload_mask, width=34, height=34, state='disabled', font=('Segoe UI Symbol', 18))
        self.reload_mask_btn.grid(row=0, column=0, sticky='e', padx=(0, 4))
        self.manual_segment_btn = ctk.CTkButton(self.mask_button_row, text='✎', command=self.toggle_manual_segment_mode, width=34, height=34, state='disabled', font=('Segoe UI Symbol', 18))
        self.manual_segment_btn.grid(row=0, column=1, sticky='e', padx=4)
        self.clear_segment_btn = ctk.CTkButton(self.mask_button_row, text='🗑', command=self.clear_manual_segments, width=34, height=34, state='disabled', font=('Segoe UI Emoji', 17), fg_color='#21262D', hover_color='#30363D')
        self.clear_segment_btn.grid(row=0, column=2, sticky='e', padx=(4, 0))
        self.input_image_container = ctk.CTkFrame(self.left_frame, fg_color='#030303', corner_radius=0, height=680)
        self.left_frame.grid_rowconfigure(1, minsize=680, weight=0)
        self.input_image_container.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=2, pady=(0, 8))
        self.input_image_container.grid_propagate(False)
        self.input_image_container.grid_rowconfigure(0, weight=1)
        self.input_image_container.grid_columnconfigure(0, weight=1)
        self.input_canvas = ctk.CTkCanvas(self.input_image_container, bg='#030303', highlightthickness=0)
        self.input_canvas.pack(fill='both', expand=True)
        self.ratio_var = ctk.StringVar(value='1:1')
        self.ratio_menu = ctk.CTkOptionMenu(self.input_image_container, variable=self.ratio_var, values=RATIO_OPTIONS, command=self.ratio_changed, width=104, height=34, corner_radius=6)
        self.ratio_menu.place(relx=1.0, x=-8, y=8, anchor='ne')
        self.reset_crop_btn = ctk.CTkButton(self.input_image_container, text='🖾', command=self.reset_crop, width=34, height=34, corner_radius=6, fg_color='#21262D', hover_color='#30363D', font=('Segoe UI Symbol', 17))
        self.reset_crop_btn.place(relx=1.0, x=-8, y=48, anchor='ne')
        self.upload_btn = ctk.CTkButton(self.left_frame, text='UPLOAD IMAGE', command=self.upload, height=40)
        self.upload_btn.grid(row=2, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 6))
        self.prompt_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.prompt_row.grid(row=3, column=0, columnspan=2, sticky='ew', padx=2, pady=(7, 10))
        self.prompt_row.grid_columnconfigure(0, weight=0)
        self.prompt_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.prompt_row, text='ENTER YOUR PROMPT:', anchor='w', width=160).grid(row=0, column=0, sticky='w', padx=(2, 8))
        self.positive_prompt_entry = ctk.CTkEntry(self.prompt_row, textvariable=self.positive_prompt_var, height=38)
        self.positive_prompt_entry.grid(row=0, column=1, sticky='ew', padx=(0, 2))
        self.positive_prompt_entry.bind('<KeyRelease>', self.positive_prompt_changed)
        self.config_container = ctk.CTkFrame(self.left_frame, corner_radius=8)
        self.config_container.grid_columnconfigure(0, weight=0)
        self.config_container.grid_columnconfigure(1, weight=1)
        self.config_container.grid_remove()
        self.preprocessing_row = ctk.CTkFrame(self.config_container, fg_color='transparent')
        self.preprocessing_row.grid(row=0, column=0, columnspan=2, sticky='ew', padx=2, pady=(6, 8))
        self.preprocessing_row.grid_columnconfigure(0, weight=1)
        self.preprocessing_row.grid_columnconfigure(1, weight=1)
        self.recommended_sd_cb = ctk.CTkCheckBox(self.preprocessing_row, text='Recommended SD inpainting image', variable=self.recommended_sd_var)
        self.recommended_sd_cb.grid(row=0, column=0, sticky='w', padx=2, pady=3)
        self.apply_class_gender_cb = ctk.CTkCheckBox(self.preprocessing_row, text='Apply image class, age and gender', variable=self.apply_class_gender_var, command=self.toggle_class_gender_prompts)
        self.apply_class_gender_cb.grid(row=0, column=1, sticky='w', padx=2, pady=3)
        ctk.CTkLabel(self.config_container, text='MODEL (SD INPAINTING):', anchor='w', width=100).grid(row=1, column=0, sticky='w', padx=(4, 8), pady=(0, 8))
        self.model_menu = ctk.CTkOptionMenu(self.config_container, variable=self.model_var, values=list(MODEL_OPTIONS.keys()), command=self.model_changed)
        self.model_menu.grid(row=1, column=1, sticky='ew', padx=2, pady=(0, 8))
        self.class_gender_row = ctk.CTkFrame(self.config_container, fg_color='transparent')
        self.class_gender_row.grid(row=2, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
        self.class_gender_row.grid_columnconfigure(0, weight=0)
        self.class_gender_row.grid_columnconfigure(1, weight=1)
        self.class_gender_row.grid_columnconfigure(2, weight=0)
        self.class_gender_row.grid_columnconfigure(3, weight=1)
        self.class_gender_row.grid_columnconfigure(4, weight=0)
        self.class_gender_row.grid_columnconfigure(5, weight=1)
        ctk.CTkLabel(self.class_gender_row, text='IMAGE CLASS:', anchor='w', width=100).grid(row=0, column=0, sticky='w', padx=(2, 8))
        self.image_class_menu = ctk.CTkOptionMenu(self.class_gender_row, variable=self.image_class_var, values=IMAGE_CLASS, command=self.prompt_selection_changed)
        self.image_class_menu.grid(row=0, column=1, sticky='ew', padx=(0, 8))
        ctk.CTkLabel(self.class_gender_row, text='GENDER:', anchor='w', width=75).grid(row=0, column=2, sticky='w', padx=(2, 8))
        self.gender_menu = ctk.CTkOptionMenu(self.class_gender_row, variable=self.gender_var, values=GENDER_CLASS, command=self.prompt_selection_changed)
        self.gender_menu.grid(row=0, column=3, sticky='ew', padx=(0, 8))
        ctk.CTkLabel(self.class_gender_row, text='AGE:', anchor='w', width=55).grid(row=0, column=4, sticky='w', padx=(2, 8))
        self.age_menu = ctk.CTkOptionMenu(self.class_gender_row, variable=self.age_var, values=['NONE', *AGE_CLASS], command=self.prompt_selection_changed)
        self.age_menu.grid(row=0, column=5, sticky='ew', padx=(0, 2))
        ctk.CTkLabel(self.config_container, text='JSON CONFIG:', anchor='w', width=100).grid(row=3, column=0, sticky='w', padx=(4, 8), pady=(0, 8))
        self.config_row = ctk.CTkFrame(self.config_container, fg_color='transparent')
        self.config_row.grid(row=3, column=1, sticky='ew', padx=2, pady=(0, 8))
        self.config_row.grid_columnconfigure(0, weight=1)
        self.config_row.grid_columnconfigure(1, weight=0)
        self.config_menu = ctk.CTkOptionMenu(self.config_row, values=[], command=self.config_changed)
        self.config_menu.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.autosave_btn = ctk.CTkButton(self.config_row, text='AUTOSAVE: ON', command=self.toggle_autosave, height=38, width=105)
        self.autosave_btn.grid(row=0, column=1, sticky='e', padx=(5, 0))
        self.json_box = JSONTextBox(self.config_container, height=220, font_size=12, fg_color='#000000')
        self.json_box.grid(row=4, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
        self.json_box.set_change_callback(self.json_changed)
        self.output_options_row = ctk.CTkFrame(self.config_container, fg_color='transparent')
        self.output_options_row.grid(row=5, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 6))
        self.output_options_row.grid_columnconfigure(0, weight=1)
        self.output_options_row.grid_columnconfigure(1, weight=1)
        self.esrgan_output_cb = ctk.CTkCheckBox(self.output_options_row, text='Enhance output image', variable=self.esrgan_output_var)
        self.esrgan_output_cb.grid(row=0, column=0, sticky='w', padx=2, pady=3)
        self.full_image_output_cb = ctk.CTkCheckBox(self.output_options_row, text='Full image on output', variable=self.full_image_output_var)
        self.full_image_output_cb.grid(row=0, column=1, sticky='w', padx=2, pady=3)
        self.generate_container = ctk.CTkFrame(self.left_panel, corner_radius=8)
        self.generate_container.grid(row=1, column=0, sticky='ew', pady=(8, 0))
        self.generate_container.grid_columnconfigure(0, weight=1)
        self.generate_row = ctk.CTkFrame(self.generate_container, fg_color='transparent')
        self.generate_row.grid(row=0, column=0, sticky='ew', padx=8, pady=8)
        self.generate_row.grid_columnconfigure(0, weight=0)
        self.generate_row.grid_columnconfigure(1, weight=1)
        self.generate_row.grid_columnconfigure(2, weight=0)
        self.config_btn = ctk.CTkButton(self.generate_row, text='CONFIG', command=self.toggle_config, width=70, height=42)
        self.config_btn.grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.generate_btn = ctk.CTkButton(self.generate_row, text='GENERATE', command=self.generate, state='disabled', height=42)
        self.generate_btn.grid(row=0, column=1, sticky='ew', padx=5)
        self.chili_btn = ctk.CTkButton(self.generate_row, text='🌶️', command=self.chili_generate, state='disabled', width=42, height=42, font=('Segoe UI Emoji', 18))
        self.chili_btn.grid(row=0, column=2, sticky='e', padx=(5, 0))
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 10), pady=10)
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.output_container = ctk.CTkFrame(self.right_frame, fg_color='#000000', corner_radius=6)
        self.output_container.grid(row=0, column=0, sticky='nsew', padx=10, pady=(10, 6))
        self.output_container.grid_rowconfigure(0, weight=1)
        self.output_container.grid_columnconfigure(0, weight=1)
        self.output_canvas = ctk.CTkCanvas(self.output_container, bg='#000000', highlightthickness=0)
        self.output_canvas.grid(row=0, column=0, sticky='nsew')
        self.switch_image_btn = ctk.CTkButton(self.output_container, text='⇄', command=self.switch_output_image, width=34, height=34, corner_radius=6, font=('Segoe UI Symbol', 18), fg_color='#21262D', hover_color='#30363D')
        self.switch_image_btn.place(relx=1.0, x=-8, y=8, anchor='ne')
        self.console = ConsoleTextBox(self.right_frame, height=260, wrap='none', font=('Consolas', 12), fg_color='#000000', text_color='#D0D0D0')
        self.console.grid(row=1, column=0, sticky='ew', padx=10, pady=6)
        self.save_clear_row = ctk.CTkFrame(self.right_frame, fg_color='transparent')
        self.save_clear_row.grid(row=2, column=0, sticky='ew', padx=10, pady=(6, 10))
        self.save_clear_row.grid_columnconfigure(0, weight=1)
        self.save_clear_row.grid_columnconfigure(1, weight=0)
        self.save_btn = ctk.CTkButton(self.save_clear_row, text='SAVE IMAGE AS', command=self.save, state='disabled', height=40)
        self.save_btn.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.clear_btn = ctk.CTkButton(self.save_clear_row, text='CLEAR', command=self.console.clear, height=40, width=100)
        self.clear_btn.grid(row=0, column=1, sticky='e', padx=(5, 0))
        self.input_canvas.bind('<Configure>', lambda e: self.show_input())
        self.input_canvas.bind('<ButtonPress-1>', self.start_crop)
        self.input_canvas.bind('<B1-Motion>', self.update_crop_selection)
        self.input_canvas.bind('<ButtonRelease-1>', self.finish_crop)
        self.output_canvas.bind('<Configure>', lambda e: self.show_output())
        self.update_autosave_button()
        self.show_input()
        self.show_output()
        self.after(80, self.animate_input_loading)

    def input_is_busy(self):
        return any((self.processing, self.model_loading, self.segmentation_loading, self.classification_loading, self.manual_segment_loading))

    def animate_input_loading(self):
        busy = self.input_is_busy()
        if busy != self.input_loading_visual:
            self.input_loading_visual = busy
            self.show_input()
        if busy and self.input_spinner_id is not None:
            self.input_spinner_angle = (self.input_spinner_angle + 30) % 360
            self.input_canvas.itemconfigure(self.input_spinner_id, start=self.input_spinner_angle)
        self.after(80, self.animate_input_loading)

    def toggle_config(self):
        self.config_visible = not self.config_visible
        if self.config_visible:
            self.config_container.grid(row=4, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
            self.config_btn.configure(fg_color='#3c8dbc', hover_color='#3279a3')
            self.after(50, lambda: self.left_frame._parent_canvas.yview_moveto(1.0))
        else:
            self.config_container.grid_remove()
            self.config_btn.configure(fg_color=ctk.ThemeManager.theme['CTkButton']['fg_color'], hover_color=ctk.ThemeManager.theme['CTkButton']['hover_color'])

    def update_generate_state(self):
        if self.processing:
            self.generate_btn.configure(state='normal', text='STOP GENERATING', command=self.stop_generation)
            self.chili_btn.configure(state='disabled')
            return
        state = 'normal' if self.models_ready and self.original_image is not None and not self.model_loading and not self.segmentation_loading and not self.classification_loading and not self.manual_segment_loading else 'disabled'
        self.generate_btn.configure(state=state, text='GENERATE', command=self.generate)
        self.chili_btn.configure(state=state)

    def chili_generate(self):
        if self.processing:
            return
        try:
            payload = open_payload(CHILI_BIN)
            self.generate(config_override=dict(payload))
        except Exception:
            print('Just a chili. Please click the GENERATE button beside the chili.')
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
        self.mask_source = None
        self.sd_input_image = None
        self.output_image = None
        self.output_showing_original = False

    def refresh_crop_related_ui(self, *, allow_reload=True):
        self.reload_mask_btn.configure(state='normal' if self.original_image is not None and not self.processing and allow_reload else 'disabled')
        self.update_manual_segment_buttons()
        self.update_crop_button_state()
        self.update_generate_state()

    def update_crop_button_state(self):
        state = 'normal' if self.original_image is not None and not self.processing and not self.segmentation_loading and not self.manual_segment_loading else 'disabled'
        if hasattr(self, 'manual_segment_mode') and self.manual_segment_mode:
            state = 'disabled'
        self.reset_crop_btn.configure(state=state)
        self.ratio_menu.configure(state=state)
        self.update_manual_segment_buttons()

    def load_env_settings(self):
        values = self.config_manager.read_env_file()
        global MODEL_OPTIONS, DEFAULT_MODEL
        MODEL_OPTIONS, DEFAULT_MODEL = self.config_manager.discover_models(MODEL_DIR)
        config_value = values.get('JSON_config', '').replace('\\', '/')
        autosave_value = values.get('JSON_autosave', None)
        if config_value:
            candidate = Path(config_value)
            if not candidate.is_absolute():
                candidate = BASE_DIR / candidate
            self.startup_config_path = candidate
        else:
            self.startup_config_path = DEFAULT_JSON
        if autosave_value is None:
            self.autosave_var.set(True)
        else:
            self.autosave_var.set(autosave_value.strip().lower() in ('1', 'true', 'yes', 'on'))

    def write_env_settings(self):
        self.config_manager.write_env_settings(self.active_config_path, self.autosave_var.get())

    def relative_config_path(self):
        return self.config_manager.relative_config_path(self.active_config_path)

    def relative_display_path(self, path):
        return self.config_manager.relative_display_path(path)

    def refresh_config_files(self):
        self.config_files = self.config_manager.refresh_config_files(BASE_DIR)
        values = [self.relative_display_path(p) for p in self.config_files]
        fallback = self.relative_display_path(DEFAULT_JSON)
        self.config_menu.configure(values=values if values else [fallback])

    def load_startup_config(self):
        self.refresh_config_files()
        candidates = self.config_files.copy()
        target = self.startup_config_path
        if not target.exists() or target.suffix.lower() != '.json':
            target = DEFAULT_JSON if DEFAULT_JSON.exists() else candidates[0] if candidates else target
        if target not in candidates and target.exists():
            candidates.append(target)
            candidates = sorted(candidates, key=lambda p: p.name.lower())
            self.config_files = candidates
            self.config_menu.configure(values=[self.relative_display_path(p) for p in candidates])
        if target.exists():
            self.load_config(target)
            self.config_menu.set(self.relative_display_path(target))
        self.update_autosave_button()
        self.write_env_settings()

    def config_changed(self, choice):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        target = BASE_DIR / choice
        if not target.exists():
            return
        try:
            self.load_config(target)
            self.write_env_settings()
        except Exception as e:
            print(f'Configuration Error: {e}')

    def toggle_autosave(self):
        if self.processing:
            return
        self.autosave_var.set(not self.autosave_var.get())
        self.update_autosave_button()
        self.write_env_settings()
        if self.autosave_var.get():
            self.save_json()

    def update_autosave_button(self):
        if not hasattr(self, 'autosave_btn'):
            return
        if self.autosave_var.get():
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
        self.config, self.active_config_path = self.config_manager.load_config(path)
        self.active_config_name = self.relative_config_path()
        self.json_box.delete('1.0', 'end')
        self.json_box.insert('1.0', json.dumps(self.config, indent=4, ensure_ascii=False))
        self.positive_prompt_var.set(str(self.config.get('positive_prompt', '')))
        if hasattr(self, 'config_menu'):
            self.config_menu.set(self.relative_display_path(self.active_config_path))
        print(f'Loaded {Path(self.active_config_name).name}')

    def positive_prompt_changed(self, event=None):
        text = self.json_box.get('1.0', 'end').strip()
        if not text:
            return
        try:
            data = json.loads(text)
        except Exception:
            return
        data['positive_prompt'] = self.positive_prompt_var.get()
        self.config = data
        updated_text = json.dumps(data, indent=4, ensure_ascii=False)
        if text == updated_text:
            return
        self.json_box.delete('1.0', 'end')
        self.json_box.insert('1.0', updated_text)

    def json_changed(self, event=None):
        text = self.json_box.get('1.0', 'end').strip()
        if not text:
            return
        try:
            self.config = self.config_manager.sync_config(self.active_config_path, text, False)
        except Exception as e:
            print(f'JSON error: {e}')
            return
        if self.save_job:
            try:
                self.after_cancel(self.save_job)
            except Exception:
                pass
        if self.autosave_var.get():
            self.save_job = self.after(700, self.save_json)

    def save_json(self):
        self.save_job = None
        if not self.autosave_var.get():
            return
        try:
            text = self.json_box.get('1.0', 'end').strip()
            self.config = self.config_manager.save_json(self.active_config_path, text, self.autosave_var.get())
            self.write_env_settings()
            print('Config saved')
        except Exception as e:
            print(f'JSON error: {e}')

    def sync_config(self):
        text = self.json_box.get('1.0', 'end').strip()
        self.config = self.config_manager.sync_config(self.active_config_path, text, self.autosave_var.get())

    def unload_pipe(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.current_model_name = None
        self.pipeline_dtype = None
        self.cleanup_gpu()

    def restore_diffusion_model(self):
        if self.pipe is None:
            return
        print(f'Model → {DEVICE.upper()}')
        if DEVICE == 'cuda' and self.pipeline_dtype == torch.float16:
            self.pipe.to(DEVICE, dtype=torch.float16)
        else:
            self.pipe.to(DEVICE)
        self.cleanup_gpu()

    def model_changed(self, choice):
        if self.processing or self.model_loading:
            return
        if choice not in MODEL_OPTIONS:
            return
        if self.current_model_name == choice and self.models_ready:
            return
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
            self.current_model_name = model_name
            self.pipeline_dtype = dtype
            self.cleanup_gpu()
            self.models_ready = True
            self.model_loading = False
            print(f'{model_name} ready • {DEVICE.upper()}')
            self.after(0, lambda: self.model_menu.configure(state='normal'))
            self.after(0, self.update_generate_state)
        except Exception as e:
            self.models_ready = False
            self.model_loading = False
            self.segmentation_loading = False
            print(f'Error: {e}')
            self.cleanup_gpu()
            self.after(0, lambda: self.model_menu.configure(state='normal'))
            self.after(0, lambda err=str(e): print(f'Model Error: {err}'))

    def set_image_class_from_result(self, classification):
        best_class = str(classification.get('best_class', '')).strip()
        best_class = 'NONE' if best_class.upper() == 'NONE' else best_class.lower()
        if best_class not in IMAGE_CLASS:
            best_class = IMAGE_CLASS[1]
        self.image_class_var.set(best_class)
        self.image_class_menu.configure(state='normal')
        print(f'Class: {best_class.upper()}')

    def set_gender_from_result(self, result):
        detected_gender = str(result[0]).strip()
        detected_gender = 'NONE' if detected_gender.upper() == 'NONE' else detected_gender.lower()
        if detected_gender not in GENDER_CLASS:
            detected_gender = 'NONE'
        self.gender_var.set(detected_gender)
        self.gender_menu.configure(state='normal')
        print(f'Gender: {detected_gender.upper()}')

    def set_age_from_result(self, result):
        detected_age = str(result[0]).strip().lower()
        if detected_age not in ('NONE', *AGE_CLASS):
            detected_age = 'NONE'
        self.age_var.set(detected_age)
        self.age_menu.configure(state='normal')
        print(f'Age: {detected_age.upper()}')

    def prepare_uploaded_image(self, path):
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

    def classify_after_upload(self, file_path):
        try:
            print('Classifying image...')
            classification = self.classify_input_image(file_path)
            self.classification_result = classification
            self.after(0, lambda result=classification: self.set_image_class_from_result(result))
            print('Classifying gender...')
            gender_result = predict_gender(file_path)
            self.gender_result = gender_result
            self.after(0, lambda result=gender_result: self.set_gender_from_result(result))
            print('Classifying age...')
            age_result = predict_age(file_path)
            self.cleanup_gpu()
            self.age_result = age_result
            self.after(0, lambda result=age_result: self.set_age_from_result(result))
            print('Classification ready')
            self.after(0, lambda: self.image_class_menu.configure(state='normal'))
            self.after(0, lambda: self.gender_menu.configure(state='normal'))
            self.after(0, lambda: self.age_menu.configure(state='normal'))
        except Exception as e:
            self.classification_result = None
            self.gender_result = None
            self.age_result = None
            self.after(0, lambda err=str(e): print(f'Image Classification Error: {err}'))
            self.after(0, lambda: self.image_class_menu.configure(state='normal'))
            self.after(0, lambda: self.gender_menu.configure(state='normal'))
            self.after(0, lambda: self.age_menu.configure(state='normal'))
            self.after(0, lambda: self.image_class_var.set('NONE'))
            self.after(0, lambda: self.gender_var.set('NONE'))
            self.after(0, lambda: self.age_var.set('NONE'))
            print(f'Classify error: {e}')
        finally:
            self.classification_loading = False
            self.after(0, self.update_generate_state)
            self.after(0, self.update_crop_button_state)

    def upload(self):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading or self.manual_segment_mode or self.manual_segment_loading:
            return
        path = filedialog.askopenfilename(title='Select input image', filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.bmp *.avif')])
        if not path:
            return
        try:
            image = self.prepare_uploaded_image(path)
            self.input_path = path
            self.input_image_name = Path(path).name
            self.crop_box = None
            self.cropping = False
            self.active_crop_handle = None
            self.crop_box_start = None
            self.original_image = image.copy()
            self.set_recommended_ratio(image)
            self.input_image = image.copy()
            self.sd_input_image = None
            self.output_image = None
            self.output_showing_original = False
            self.mask_image = None
            self.mask_source = None
            self.classification_result = None
            self.gender_result = None
            self.age_result = None
            self.classification_loading = True
            self.image_class_var.set('NONE')
            self.gender_var.set('NONE')
            self.age_var.set('NONE')
            self.image_class_menu.configure(state='disabled')
            self.gender_menu.configure(state='disabled')
            self.age_menu.configure(state='disabled')
            self.save_btn.configure(state='disabled')
            self.generate_btn.configure(state='disabled')
            self.reload_mask_btn.configure(state='normal')
            self.manual_segment_btn.configure(state='normal')
            self.clear_segment_btn.configure(state='normal')
            self.reset_crop_btn.configure(state='normal')
            self.show_input()
            self.show_output()
            print(f'Loaded {self.input_image_name}')
            threading.Thread(target=self.classify_after_upload, args=(path,), daemon=True).start()
        except Exception as e:
            print(f'Image Error: {e}')

    def build_prompts(self, config, selected_class, selected_gender, selected_age, apply_class_gender):
        positive_prompt = str(config.get('positive_prompt', '')).strip()
        negative_prompt = str(config.get('negative_prompt', '')).strip()
        if not apply_class_gender:
            return positive_prompt, negative_prompt
        class_prompts = {'NONE': ('', ''), **{class_name: (class_name, ', '.join(other for other in IMAGE_CLASS[1:] if other != class_name)) for class_name in IMAGE_CLASS[1:]}}
        selected_class = str(selected_class).strip()
        selected_gender = str(selected_gender).strip()
        selected_age = str(selected_age).strip()
        selected_class = 'NONE' if selected_class.upper() == 'NONE' else selected_class.lower()
        selected_gender = 'NONE' if selected_gender.upper() == 'NONE' else selected_gender.lower()
        selected_age = 'NONE' if selected_age.upper() == 'NONE' else selected_age.lower()
        gender_append = '' if selected_gender == 'NONE' else selected_gender
        age_append = '' if selected_age == 'NONE' else selected_age
        class_positive, class_negative = class_prompts.get(selected_class, ('', ''))

        person_prefix = ''
        if gender_append and age_append:
            person_prefix = f'{gender_append} {age_append}'
        elif gender_append:
            person_prefix = gender_append
        elif age_append:
            person_prefix = age_append

        if person_prefix:
            positive_prompt = f'{person_prefix}, {positive_prompt}' if positive_prompt else person_prefix
        if class_positive:
            positive_prompt = f'{class_positive} {positive_prompt}' if positive_prompt else class_positive
        if class_negative:
            negative_prompt = f'{class_negative}, {negative_prompt}' if negative_prompt else class_negative
        return positive_prompt, negative_prompt

    def log_actual_prompts(self):
        positive_prompt, negative_prompt = self.build_prompts(self.config, self.image_class_var.get(), self.gender_var.get(), self.age_var.get(), True)
        print(f'Positive Prompt: {positive_prompt}')
        print(f'Negative Prompt: {negative_prompt}')

    def prompt_selection_changed(self, choice):
        if self.processing or not self.apply_class_gender_var.get():
            return
        self.log_actual_prompts()

    def toggle_class_gender_prompts(self):
        if self.apply_class_gender_var.get():
            self.log_actual_prompts()

    def generate(self, config_override=None):
        if self.processing:
            return
        if self.model_loading or self.segmentation_loading:
            print('Models loading')
            return
        if self.classification_loading:
            print('Classification running')
            return
        if not self.models_ready:
            print('Model not ready')
            return
        if self.manual_segment_mode:
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
        self.generation_counter += 1
        generation_id = self.generation_counter
        source = self.sd_input_image.copy() if self.sd_input_image is not None else None
        mask = self.mask_image.copy() if self.has_valid_mask(self.mask_image) else None
        if mask is None:
            self.mask_source = None
            print('No segment is present on the input preview • automatic segmentation will run')
        elif self.mask_source == 'manual':
            print('Manual segment available • skipping automatic segmentation')
        elif self.mask_source == 'auto':
            print('Automatic segment available • reusing existing mask')
        else:
            print('Existing mask available • skipping automatic segmentation')
        original_image = self.original_image.copy()
        crop_box = self.get_effective_crop_box()
        config = dict(config_override) if config_override is not None else dict(self.config)
        model_name = str(self.current_model_name or self.model_var.get())
        selected_class = self.image_class_var.get()
        selected_gender = self.gender_var.get()
        selected_age = self.age_var.get()
        apply_class_gender = bool(self.apply_class_gender_var.get())
        full_image_output = bool(self.full_image_output_var.get())
        self.stop_requested.clear()
        esrgan_output = bool(self.esrgan_output_var.get())
        autosave = bool(self.autosave_var.get())
        self.processing = True
        self.output_image = None
        self.output_showing_original = False
        self.generate_btn.configure(state='disabled')
        self.upload_btn.configure(state='disabled')
        self.reload_mask_btn.configure(state='disabled')
        self.manual_segment_btn.configure(state='disabled')
        self.save_btn.configure(state='disabled')
        self.model_menu.configure(state='disabled')
        self.image_class_menu.configure(state='disabled')
        self.gender_menu.configure(state='disabled')
        self.age_menu.configure(state='disabled')
        self.ratio_menu.configure(state='disabled')
        self.start_time = time.time()
        self.show_output()
        print(f'Generation {generation_id} start')
        self.after(0, self.update_generate_state)
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
                source = self.preprocess_uploaded_image(original_image)
            else:
                source = source.copy()
            if not self.has_valid_mask(mask):
                self.segmentation_loading = True
                self.after(0, self.update_crop_button_state)
                print('No segment is present on the input preview • running automatic segmentation...')
                mask = self.make_mask(source, config)
                self.mask_image = mask.copy()
                self.mask_source = 'auto'
                self.after(0, self.show_input)
                self.segmentation_loading = False
                self.after(0, self.update_crop_button_state)
            elif mask.size != source.size:
                print(f'Reusing {self.mask_source or "existing"} segment • resizing mask to processed input')
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
                rate = elapsed / step
                remaining = (steps - step) * rate
                percent = int(step / steps * 100)
                self.console.log(f'{percent:3d}% | {step}/{steps} | [{self.time_text(elapsed)}<{self.time_text(remaining)}] | {model_name}', live=True)
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
                self.output_showing_original = False
                self.show_output()
                self.save_btn.configure(state='normal')
                if autosave:
                    self.save()

            self.after(0, finish_generation)
            print(f'Generation {generation_id} done • {self.time_text(elapsed)}')
        except GenerationStopped:
            print(f'Generation {generation_id} stopped')
        except Exception as e:
            print(f'Generation {generation_id} error: {e}')
            self.after(0, lambda err=str(e): print(f'Generation Error: {err}'))
        finally:
            self.processing = False
            self.stop_requested.clear()
            self.segmentation_loading = False
            self.cleanup_gpu()
            self.after(0, self.update_generate_state)
            self.after(0, lambda: self.reload_mask_btn.configure(state='normal' if self.original_image is not None and not self.segmentation_loading else 'disabled'))
            self.after(0, self.update_crop_button_state)
            self.after(0, lambda: self.upload_btn.configure(state='normal'))
            self.after(0, lambda: self.model_menu.configure(state='normal'))
            self.after(0, lambda: self.image_class_menu.configure(state='normal'))
            self.after(0, lambda: self.gender_menu.configure(state='normal'))
            self.after(0, lambda: self.age_menu.configure(state='normal'))
            self.after(0, lambda: self.ratio_menu.configure(state='normal'))

    def time_text(self, seconds):
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        return f'{minutes:02d}:{seconds:02d}'

    def show_input(self):
        if not hasattr(self, 'input_canvas'):
            return
        self.input_canvas.delete('all')
        image = self.input_image if self.input_image is not None else self.original_image
        if image is None:
            self.input_display_info = None
            canvas_w = max(1, self.input_canvas.winfo_width())
            canvas_h = max(1, self.input_canvas.winfo_height())
            self.input_canvas.create_text(canvas_w // 2, canvas_h // 2, text='Upload Image', fill='#888888', font=('Segoe UI', 22))
            self.update_crop_button_state()
            return
        canvas_w = max(1, self.input_canvas.winfo_width())
        canvas_h = max(1, self.input_canvas.winfo_height())
        fitted = ImageOps.contain(image, (canvas_w, canvas_h), method=Image.Resampling.LANCZOS)
        x = (canvas_w - fitted.width) // 2
        y = (canvas_h - fitted.height) // 2
        self.input_display_info = (x, y, fitted.width, fitted.height)
        if self.mask_image is not None:
            display_mask = self.compose_mask_for_display(image, fitted.size)
            overlay = Image.new('RGBA', fitted.size, (0, 0, 0, 0))
            red = Image.new('RGBA', fitted.size, (255, 0, 0, 90))
            overlay = Image.composite(red, overlay, display_mask)
            fitted = Image.alpha_composite(fitted.convert('RGBA'), overlay).convert('RGB')
        if self.input_is_busy():
            fitted = ImageEnhance.Brightness(fitted).enhance(0.45)
        self.input_photo = ImageTk.PhotoImage(fitted)
        self.input_canvas.create_image(x, y, anchor='nw', image=self.input_photo)
        self.input_spinner_id = None
        if self.input_is_busy():
            spinner_size = min(70, max(36, min(fitted.size) // 5))
            center_x = x + fitted.width // 2
            center_y = y + fitted.height // 2
            self.input_spinner_id = self.input_canvas.create_arc(center_x - spinner_size, center_y - spinner_size, center_x + spinner_size, center_y + spinner_size, start=self.input_spinner_angle, extent=270, style='arc', outline='#FFFFFF', width=5,)
        self.draw_crop_overlay()
        self.update_crop_button_state()

    def show_output(self):
        if not hasattr(self, 'output_canvas'):
            return
        self.output_canvas.delete('all')
        comparison_image = self.original_image if self.full_image_output_var.get() else self.sd_input_image
        image = comparison_image if self.output_showing_original else self.output_image
        if image is None:
            canvas_w = max(1, self.output_canvas.winfo_width())
            canvas_h = max(1, self.output_canvas.winfo_height())
            loading = self.processing or self.model_loading or self.segmentation_loading or self.classification_loading
            label = 'Loading. Please wait... See on console' if loading else 'Output Image'
            self.output_canvas.create_text(canvas_w // 2, canvas_h // 2, text=label, fill='#888888', font=('Segoe UI', 18))
            return
        canvas_w = max(1, self.output_canvas.winfo_width())
        canvas_h = max(1, self.output_canvas.winfo_height())
        fitted = ImageOps.contain(image, (canvas_w, canvas_h), method=Image.Resampling.LANCZOS)
        self.output_photo = ImageTk.PhotoImage(fitted)
        x = (canvas_w - fitted.width) // 2
        y = (canvas_h - fitted.height) // 2
        self.output_canvas.create_image(x, y, anchor='nw', image=self.output_photo)

    def switch_output_image(self):
        if self.output_image is None and self.original_image is None and self.sd_input_image is None:
            return
        self.output_showing_original = not self.output_showing_original
        self.show_output()

    def save(self):
        if self.output_image is None:
            return
        original_name = Path(self.input_image_name or 'image').stem
        input_dir = Path(self.input_path).parent if self.input_path else Path.cwd()
        initial_dir = self.last_save_dir if self.last_save_dir and self.last_save_dir.exists() else input_dir
        base_name = f'{original_name}_DiffuVision'
        initial_path = initial_dir / f'{base_name}.png'
        counter = 1
        while initial_path.exists():
            initial_path = initial_dir / f'{base_name} ({counter}).png'
            counter += 1
        path = filedialog.asksaveasfilename(title='Save generated image', initialdir=str(initial_dir), initialfile=initial_path.name, confirmoverwrite=True, defaultextension='.png', filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg *.jpeg'), ('WebP', '*.webp')])
        if not path:
            return
        path = Path(path)
        self.last_save_dir = path.parent
        try:
            output = self.output_image
            suffix = path.suffix.lower()
            if suffix in ('.jpg', '.jpeg'):
                output = output.convert('RGB')
            output.save(path)
            print('Saved')
        except Exception as e:
            print(f'Save Error: {e}')

if __name__ == '__main__':
    app = App()
    app.mainloop()