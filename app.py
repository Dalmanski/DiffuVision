import os, sys, json, time, threading, gc, math
from pathlib import Path
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageTk, ImageFilter, ImageOps
import numpy as np, torch, cv2
import torchvision.transforms.functional as TF
sys.modules.setdefault('torchvision.transforms.functional_tensor', TF)
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from modules.segdinosam2 import SegDinoSAM2
from modules.selectsegsam2 import SAM2Segmenter
from modules.imgclass import REAL, ANIME, THREE_D, CARTOON, classify_image
from components.json_textbox import JSONTextBox
from components.console_textbox import ConsoleTextBox, create_redirects
from utils.config_manager import ConfigManager
from modules import gender as gender_module
from model.upscale_img import enhance

BASE_DIR = Path(__file__).resolve().parent
ctk.set_appearance_mode('system')
ctk.set_default_color_theme(str(BASE_DIR / 'themes' / 'red.json'))
MODEL_DIR = BASE_DIR / 'model'
DEFAULT_JSON = BASE_DIR / 'data/diffusion_config/default.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_OPTIONS = {}
DEFAULT_MODEL = ''
MAX_SIDE = 768
RECOMMENDED_RATIO_SIZES = {'1:1': (512, 512), '4:3': (768, 576), '3:2': (768, 512), '16:9': (768, 432), '5:4': (640, 512), '4:5': (512, 640), '3:4': (576, 768), '2:3': (512, 768), '9:16': (432, 768)}
RATIO_OPTIONS = ['1:1', '4:3', '3:2', '16:9', '5:4', '4:5', '3:4', '2:3', '9:16']
OUTPUT_TARGET = 1080
IMAGE_CLASSES = [REAL, ANIME, THREE_D, CARTOON]

class GenerationStopped(Exception):
    pass

class App(ctk.CTk):
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
        self.generation_counter = 0
        self.mask_image = None
        self.mask_source = None
        self.classification_result = None
        self.gender_result = None
        self.classification_loading = False
        self.processing = False
        self.models_ready = False
        self.model_loading = False
        self.segmentation_loading = False
        self.start_time = 0
        self.save_job = None
        self.input_photo = None
        self.output_photo = None
        self.crop_box = None
        self.crop_box_start = None
        self.input_display_info = None
        self.cropping = False
        self.active_crop_handle = None
        self.resize_var = ctk.BooleanVar(value=True)
        self.esrgan_input_var = ctk.BooleanVar(value=False)
        self.esrgan_output_var = ctk.BooleanVar(value=False)
        self.full_image_output_var = ctk.BooleanVar(value=True)
        self.apply_class_gender_var = ctk.BooleanVar(value=False)
        self.stop_requested = threading.Event()
        self.autosave_var = ctk.BooleanVar(value=True)
        self.model_var = ctk.StringVar(value='')
        self.image_class_var = ctk.StringVar(value=REAL)
        self.gender_var = ctk.StringVar(value='neutral')
        self.config_files = []
        self.active_config_name = 'data/diffusion_config/default.json'
        self.active_config_path = DEFAULT_JSON
        self.config = {}
        self.config_manager = ConfigManager(BASE_DIR, DEFAULT_JSON)
        self.env_path = self.config_manager.env_path
        self.output_showing_original = False
        self.load_env_settings()
        self.model_var.set(DEFAULT_MODEL)
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        self.ui()
        self.stdout_redirect, self.stderr_redirect = create_redirects(self.console)
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stderr_redirect
        self.after(100, self.maximize)
        self.refresh_config_files()
        self.load_startup_config()
        if self.model_var.get():
            threading.Thread(target=self.load_models, args=(self.model_var.get(),), daemon=True).start()

    def maximize(self):
        try:
            self.state('zoomed')
        except Exception:
            pass

    def ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.left_frame = ctk.CTkScrollableFrame(self)
        self.left_frame.grid(row=0, column=0, sticky='nsew', padx=(10, 5), pady=10)
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
        self.clear_segment_btn = ctk.CTkButton(self.mask_button_row, text='🗑', command=self.clear_manual_segments, width=34, height=34, state='disabled', font=('Segoe UI Symbol', 17), fg_color='#21262D', hover_color='#30363D')
        self.clear_segment_btn.grid(row=0, column=2, sticky='e', padx=(4, 0))
        self.input_image_container = ctk.CTkFrame(self.left_frame, fg_color='#030303', corner_radius=0, height=700)
        self.left_frame.grid_rowconfigure(1, minsize=700, weight=0)
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
        self.auto_fit_crop_btn = ctk.CTkButton(self.input_image_container, text='⿻', command=self.auto_fit_crop_to_person, width=34, height=34, corner_radius=6, fg_color='#21262D', hover_color='#30363D', font=('Segoe UI Symbol', 17))
        self.auto_fit_crop_btn.place(relx=1.0, x=-8, y=88, anchor='ne')
        self.upload_btn = ctk.CTkButton(self.left_frame, text='UPLOAD IMAGE', command=self.upload, height=40)
        self.upload_btn.grid(row=2, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 6))
        self.preprocessing_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.preprocessing_row.grid(row=3, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
        self.preprocessing_row.grid_columnconfigure(0, weight=1)
        self.preprocessing_row.grid_columnconfigure(1, weight=1)
        self.resize_cb = ctk.CTkCheckBox(self.preprocessing_row, text='Auto resize original image for recommended SD inpainting', variable=self.resize_var)
        self.resize_cb.grid(row=0, column=0, sticky='w', padx=2, pady=3)
        self.esrgan_input_cb = ctk.CTkCheckBox(self.preprocessing_row, text='Enhance original image', variable=self.esrgan_input_var)
        self.esrgan_input_cb.grid(row=0, column=1, sticky='w', padx=2, pady=3)
        ctk.CTkLabel(self.left_frame, text='MODEL (SD INPAINTING):', anchor='w', width=100).grid(row=4, column=0, sticky='w', padx=(4, 8), pady=(0, 8))
        self.model_menu = ctk.CTkOptionMenu(self.left_frame, variable=self.model_var, values=list(MODEL_OPTIONS.keys()), command=self.model_changed)
        self.model_menu.grid(row=4, column=1, sticky='ew', padx=2, pady=(0, 8))
        self.class_gender_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.class_gender_row.grid(row=5, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
        self.class_gender_row.grid_columnconfigure(0, weight=0)
        self.class_gender_row.grid_columnconfigure(1, weight=1)
        self.class_gender_row.grid_columnconfigure(2, weight=0)
        self.class_gender_row.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(self.class_gender_row, text='IMAGE CLASS:', anchor='w', width=100).grid(row=0, column=0, sticky='w', padx=(2, 8))
        self.image_class_menu = ctk.CTkOptionMenu(self.class_gender_row, variable=self.image_class_var, values=IMAGE_CLASSES)
        self.image_class_menu.grid(row=0, column=1, sticky='ew', padx=(0, 8))
        ctk.CTkLabel(self.class_gender_row, text='GENDER:', anchor='w', width=75).grid(row=0, column=2, sticky='w', padx=(2, 8))
        self.gender_menu = ctk.CTkOptionMenu(self.class_gender_row, variable=self.gender_var, values=['male', 'female', 'neutral'])
        self.gender_menu.grid(row=0, column=3, sticky='ew', padx=(0, 2))
        ctk.CTkLabel(self.left_frame, text='JSON CONFIG:', anchor='w', width=100).grid(row=6, column=0, sticky='w', padx=(4, 8), pady=(0, 8))
        self.config_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.config_row.grid(row=6, column=1, sticky='ew', padx=2, pady=(0, 8))
        self.config_row.grid_columnconfigure(0, weight=1)
        self.config_row.grid_columnconfigure(1, weight=0)
        self.config_menu = ctk.CTkOptionMenu(self.config_row, values=[], command=self.config_changed)
        self.config_menu.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.autosave_btn = ctk.CTkButton(self.config_row, text='AUTOSAVE: ON', command=self.toggle_autosave, height=38, width=105)
        self.autosave_btn.grid(row=0, column=1, sticky='e', padx=(5, 0))
        self.json_box = JSONTextBox(self.left_frame, height=220, font_size=12, fg_color='#000000')
        self.json_box.grid(row=7, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 8))
        self.json_box.set_change_callback(self.json_changed)
        self.output_options_row = ctk.CTkFrame(self.left_frame, fg_color='transparent')
        self.output_options_row.grid(row=8, column=0, columnspan=2, sticky='ew', padx=2, pady=(0, 0))
        self.output_options_row.grid_columnconfigure(0, weight=1)
        self.output_options_row.grid_columnconfigure(1, weight=1)
        self.output_options_row.grid_columnconfigure(2, weight=1)
        self.esrgan_output_cb = ctk.CTkCheckBox(self.output_options_row, text='Enhance output image', variable=self.esrgan_output_var)
        self.esrgan_output_cb.grid(row=0, column=0, sticky='w', padx=2, pady=3)
        self.full_image_output_cb = ctk.CTkCheckBox(self.output_options_row, text='Full image on output', variable=self.full_image_output_var)
        self.full_image_output_cb.grid(row=0, column=1, sticky='w', padx=2, pady=3)
        self.apply_class_gender_cb = ctk.CTkCheckBox(self.output_options_row, text='Apply image class and gender', variable=self.apply_class_gender_var)
        self.apply_class_gender_cb.grid(row=0, column=2, sticky='w', padx=2, pady=3)
        self.generate_btn = ctk.CTkButton(self.left_frame, text='GENERATE', command=self.generate, state='disabled', height=42)
        self.generate_btn.grid(row=9, column=0, columnspan=2, sticky='ew', padx=2, pady=(12, 8))
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

    def update_generate_state(self):
        if self.processing:
            self.generate_btn.configure(state='normal', text='STOP GENERATING', command=self.stop_generation)
            return
        state = 'normal' if self.models_ready and self.original_image is not None and not self.model_loading and not self.segmentation_loading and not self.classification_loading and not self.manual_segment_loading else 'disabled'
        self.generate_btn.configure(state=state, text='GENERATE', command=self.generate)

    def stop_generation(self):
        if not self.processing:
            return
        self.stop_requested.set()
        self.console.log('Stopping...')
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
        self.auto_fit_crop_btn.configure(state=state)
        self.ratio_menu.configure(state=state)
        self.update_manual_segment_buttons()

    def update_manual_segment_buttons(self):
        if not hasattr(self, 'manual_segment_btn'):
            return
        base_state = 'normal' if self.original_image is not None and not self.processing and not self.model_loading and not self.segmentation_loading and not self.classification_loading and not self.manual_segment_loading else 'disabled'
        self.manual_segment_btn.configure(state=base_state)
        self.clear_segment_btn.configure(state='normal' if self.original_image is not None else 'disabled')
        if self.manual_segment_mode:
            self.manual_segment_btn.configure(fg_color='#1f8f3a', hover_color='#176b2c')
        else:
            self.manual_segment_btn.configure(fg_color='#21262D', hover_color='#30363D')

    def toggle_manual_segment_mode(self):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        if self.original_image is None:
            return
        if self.manual_segment_mode:
            self.disable_manual_segment_mode()
            self.console.log('SAM2 point selection disabled')
            return
        try:
            self.sync_config()
        except Exception as e:
            self.console.log(f'Configuration Error: {e}')
            return
        self.manual_segment_mode = True
        self.manual_segment_loading = True
        self.manual_segment_btn.configure(state='disabled')
        self.clear_segment_btn.configure(state='normal' if self.original_image is not None else 'disabled')
        self.update_crop_button_state()
        self.console.log('Loading SAM2 point selection...')
        threading.Thread(target=self.load_manual_segmenter_worker, daemon=True).start()

    def load_manual_segmenter_worker(self):
        try:
            image = self.input_image.copy() if self.input_image is not None else self.original_image.copy()
            self.manual_segmenter = SAM2Segmenter()
            self.manual_segmenter.load_image(image)
            self.manual_selection_image = image
            self.console.log('SAM2 point selection ready • click the image to append segments')
        except Exception as e:
            self.manual_segment_mode = False
            self.console.log(f'SAM2 point selection error: {e}')
            self.after(0, lambda err=str(e): self.console.log(f'SAM2 Error: {err}'))
        finally:
            self.manual_segment_loading = False
            self.after(0, self.update_crop_button_state)
            self.after(0, self.show_input)
            self.after(0, self.update_generate_state)

    def disable_manual_segment_mode(self):
        self.manual_segment_mode = False
        self.manual_segment_loading = False
        segmenter = self.manual_segmenter
        self.manual_segmenter = None
        self.manual_selection_image = None
        if segmenter is not None:
            segmenter.close()
        self.cleanup_gpu()
        self.reload_mask_btn.configure(state='normal' if self.original_image is not None and not self.processing and not self.segmentation_loading else 'disabled')
        self.update_crop_button_state()
        self.show_input()

    def select_manual_segment(self, event):
        if not self.manual_segment_mode or self.manual_segment_loading or self.manual_segmenter is None:
            return
        if self.input_display_info is None or self.manual_selection_image is None:
            return
        x, y, width, height = self.input_display_info
        if event.x < x or event.x >= x + width or event.y < y or event.y >= y + height:
            return
        image_width, image_height = self.manual_selection_image.size
        image_x = max(0, min(image_width - 1, int((event.x - x) / width * image_width)))
        image_y = max(0, min(image_height - 1, int((event.y - y) / height * image_height)))
        self.manual_segment_loading = True
        self.update_manual_segment_buttons()
        threading.Thread(target=self.manual_segment_worker, args=(image_x, image_y), daemon=True).start()

    def manual_segment_worker(self, image_x, image_y):
        try:
            thickness = float(self.config.get('mask_outline_thickness', 0.0))
            blur = float(self.config.get('mask_blur', 0.0))
            self.console.log(f'Manual mask settings: outline={thickness:g}px, blur={blur:g}px')
            mask = self.manual_segmenter.select_point(image_x, image_y, positive=True, target_size=self.sd_input_image.size if self.sd_input_image is not None else None, crop_box=self.get_effective_crop_box(), original_size=self.original_image.size, thickness=thickness, blur=blur)
            self.mask_image = mask
            self.mask_source = 'manual'
            self.output_image = None
            self.output_showing_original = False
            self.after(0, self.show_input)
            self.after(0, self.show_output)
            self.console.log('Segment appended')
        except Exception as e:
            self.console.log(f'SAM2 selection error: {e}')
            self.after(0, lambda err=str(e): self.console.log(f'SAM2 Selection Error: {err}'))
        finally:
            self.manual_segment_loading = False
            self.after(0, self.update_crop_button_state)
            self.after(0, lambda: self.clear_segment_btn.configure(state='normal' if self.original_image is not None else 'disabled'))
            self.after(0, self.update_generate_state)

    def clear_manual_segments(self):
        if self.processing or self.manual_segment_loading:
            return
        if self.manual_segmenter is not None:
            self.manual_segmenter.clear_all_segments()
        self.reset_preview_state()
        self.generate_btn.configure(state='disabled')
        self.console.log('All manually appended segments removed')
        self.show_input()
        self.show_output()
        self.update_generate_state()

    def auto_fit_crop_to_person(self):
        if self.original_image is None:
            self.console.log('No image')
            return
        if self.processing:
            self.console.log('Generation running')
            return
        if self.model_loading:
            self.console.log('Model loading')
            return
        if self.segmentation_loading:
            self.console.log('Segmentation running')
            return
        if self.classification_loading:
            self.console.log('Classification running')
            return
        self.segmentation_loading = True
        self.update_crop_button_state()
        self.reload_mask_btn.configure(state='disabled')
        self.generate_btn.configure(state='disabled')
        self.console.log('Finding person...')
        threading.Thread(target=self.auto_fit_crop_worker, daemon=True).start()

    def auto_fit_crop_worker(self):
        try:
            image = self.original_image.copy()
            self.ensure_segmentation()
            self.load_segmentation()
            result = self.segmentation.segment(image, 'person', '', thickness=0)
            masks = getattr(result, 'masks', None)
            if masks is None and isinstance(result, dict):
                masks = result.get('masks')
            best_mask = None
            if masks is not None:
                for mask in masks:
                    array = np.asarray(mask).astype(bool)
                    if array.ndim > 2:
                        array = np.squeeze(array)
                    if array.ndim != 2 or not np.any(array):
                        continue
                    if best_mask is None or np.count_nonzero(array) > np.count_nonzero(best_mask):
                        best_mask = array
            if best_mask is None:
                mask_image = getattr(result, 'mask_image', None)
                if mask_image is None and isinstance(result, dict):
                    mask_image = result.get('mask_image')
                if mask_image is not None:
                    array = np.asarray(mask_image)
                    if array.ndim > 2:
                        array = np.squeeze(array)
                    if array.ndim == 2 and np.any(array):
                        best_mask = array > 0
            if best_mask is None or best_mask.ndim != 2 or not np.any(best_mask):
                raise RuntimeError('No person was detected for automatic crop fitting.')
            if best_mask.shape != (image.height, image.width):
                resized = Image.fromarray((best_mask.astype(np.uint8) * 255), 'L').resize(image.size, Image.Resampling.NEAREST)
                best_mask = np.asarray(resized, dtype=np.uint8) > 0
            ys, xs = np.where(best_mask)
            if xs.size == 0 or ys.size == 0:
                raise RuntimeError('No person was detected for automatic crop fitting.')
            left = xs.min() / image.width
            top = ys.min() / image.height
            right = (xs.max() + 1) / image.width
            bottom = (ys.max() + 1) / image.height
            margin_x = max(0.02, min(0.12, (right - left) * 0.08))
            margin_y = max(0.02, min(0.12, (bottom - top) * 0.08))
            proposed_crop_box = (left - margin_x, top - margin_y, right + margin_x, bottom + margin_y)
            ratio = self.get_crop_ratio()
            crop_box = self.clamp_fixed_crop_box(proposed_crop_box, ratio)
            self.after(0, lambda box=crop_box: self.apply_auto_person_crop(box))
        except Exception as e:
            self.console.log(f'Automatic crop error: {e}')
        finally:
            self.unload_segmentation()
            self.cleanup_gpu()
            self.segmentation_loading = False
            self.after(0, self.update_crop_button_state)
            self.after(0, lambda: self.reload_mask_btn.configure(state='normal' if self.original_image is not None and not self.processing else 'disabled'))
            self.after(0, self.update_generate_state)
            self.after(0, self.update_crop_button_state)

    def apply_auto_person_crop(self, crop_box):
        if self.original_image is None:
            return
        ratio = self.get_crop_ratio()
        self.crop_box = self.clamp_fixed_crop_box(crop_box, ratio)
        self.reset_preview_state()
        self.refresh_crop_related_ui()
        self.console.log('Person crop ready')
        self.console.log('Crop changed')
        self.show_input()
        self.show_output()

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
        self.config_menu.configure(values=values if values else ['data/diffusion_config/default.json'])

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
            self.console.log(f'Configuration Error: {e}')

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
        if hasattr(self, 'config_menu'):
            self.config_menu.set(self.relative_display_path(self.active_config_path))
        self.console.log(f'Loaded {Path(self.active_config_name).name}')

    def json_changed(self, event=None):
        text = self.json_box.get('1.0', 'end').strip()
        if not text:
            return
        try:
            self.config = self.config_manager.sync_config(self.active_config_path, text, False)
        except Exception as e:
            self.console.log(f'JSON error: {e}')
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
            self.console.log('Config saved')
        except Exception as e:
            self.console.log(f'JSON error: {e}')

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

    def offload_diffusion_model(self):
        self.cleanup_gpu()

    def restore_diffusion_model(self):
        if self.pipe is None:
            return
        self.console.log(f'Model → {DEVICE.upper()}')
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
            self.console.log(f'Loading {model_name}')
            pipe = StableDiffusionInpaintPipeline.from_single_file(model_id, torch_dtype=dtype, safety_checker=None, local_files_only=True)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(DEVICE)
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
            self.console.log(f'{model_name} ready • {DEVICE.upper()}')
            self.after(0, lambda: self.model_menu.configure(state='normal'))
            self.after(0, self.update_generate_state)
        except Exception as e:
            self.models_ready = False
            self.model_loading = False
            self.segmentation_loading = False
            self.console.log(f'Error: {e}')
            self.cleanup_gpu()
            self.after(0, lambda: self.model_menu.configure(state='normal'))
            self.after(0, lambda err=str(e): self.console.log(f'Model Error: {err}'))

    def set_image_class_from_result(self, classification):
        best_class = str(classification.get('best_class', '')).strip()
        if not best_class:
            best_class = REAL
        self.image_class_var.set(best_class)
        self.image_class_menu.configure(state='normal')
        self.console.log(f'Class: {best_class.upper()}')

    def set_gender_from_result(self, result):
        detected_gender = str(result[0]).strip().lower()
        if detected_gender not in ('male', 'female', 'neutral'):
            detected_gender = 'neutral'
        self.gender_var.set(detected_gender)
        self.gender_menu.configure(state='normal')
        self.console.log(f'Gender: {detected_gender.upper()}')

    def predict_gender_image(self, file_path):
        model = getattr(gender_module, 'model', None)
        if model is None:
            raise RuntimeError('gender.py model is not available.')

        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gender_module.device = target_device
        try:
            model.to(target_device)
            return gender_module.predict_gender(file_path)
        finally:
            self.cleanup_gpu()

    def prepare_uploaded_image(self, path):
        image = ImageOps.exif_transpose(Image.open(path))
        has_alpha = image.mode in ('RGBA', 'LA') or 'transparency' in image.info
        if has_alpha:
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert('RGB')
            self.console.log('Alpha background filled')
        else:
            image = image.convert('RGB')
        return image

    def classify_after_upload(self, file_path):
        try:
            self.console.log('Classifying image...')
            classification = self.classify_input_image(file_path)
            self.classification_result = classification
            self.after(0, lambda result=classification: self.set_image_class_from_result(result))
            self.console.log('Classifying gender...')
            gender_result = self.predict_gender_image(file_path)
            self.gender_result = gender_result
            self.after(0, lambda result=gender_result: self.set_gender_from_result(result))
            self.console.log('Classification ready')
            self.after(0, lambda: self.image_class_menu.configure(state='normal'))
            self.after(0, lambda: self.gender_menu.configure(state='normal'))
        except Exception as e:
            self.classification_result = None
            self.gender_result = None
            self.after(0, lambda err=str(e): self.console.log(f'Image Classification Error: {err}'))
            self.after(0, lambda: self.image_class_menu.configure(state='normal'))
            self.after(0, lambda: self.gender_menu.configure(state='normal'))
            self.after(0, lambda: self.image_class_var.set(REAL))
            self.after(0, lambda: self.gender_var.set('neutral'))
            self.console.log(f'Classify error: {e}')
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
            self.classification_loading = True
            self.image_class_var.set(REAL)
            self.gender_var.set('neutral')
            self.image_class_menu.configure(state='disabled')
            self.gender_menu.configure(state='disabled')
            self.save_btn.configure(state='disabled')
            self.generate_btn.configure(state='disabled')
            self.reload_mask_btn.configure(state='normal')
            self.manual_segment_btn.configure(state='normal')
            self.clear_segment_btn.configure(state='normal')
            self.reset_crop_btn.configure(state='normal')
            self.auto_fit_crop_btn.configure(state='normal')
            self.show_input()
            self.show_output()
            self.console.log(f'Loaded {self.input_image_name}')
            threading.Thread(target=self.classify_after_upload, args=(path,), daemon=True).start()
        except Exception as e:
            self.console.log(f'Image Error: {e}')

    def set_recommended_ratio(self, image):
        width, height = image.size
        aspect = width / height
        choice = min(RECOMMENDED_RATIO_SIZES, key=lambda key: abs(math.log(aspect / (RECOMMENDED_RATIO_SIZES[key][0] / RECOMMENDED_RATIO_SIZES[key][1]))))
        self.ratio_var.set(choice)
        self.crop_box = self.default_crop_box_for_ratio(choice)

    def default_crop_box_for_ratio(self, choice):
        if self.original_image is None:
            return 0.0, 0.0, 1.0, 1.0
        ratio = float(choice.split(':')[0]) / float(choice.split(':')[1])
        image_width, image_height = self.original_image.size
        image_aspect = image_width / image_height
        if image_aspect > ratio:
            crop_height = 1.0
            crop_width = ratio / image_aspect
        else:
            crop_width = 1.0
            crop_height = image_aspect / ratio
        left = (1.0 - crop_width) / 2.0
        top = (1.0 - crop_height) / 2.0
        return left, top, left + crop_width, top + crop_height

    def resize_image(self, image):
        image = image.convert('RGB')
        w, h = image.size
        ratio = self.ratio_var.get().strip().upper()
        if ratio in RECOMMENDED_RATIO_SIZES:
            nw, nh = RECOMMENDED_RATIO_SIZES[ratio]
        else:
            longest = max(w, h)
            if longest <= MAX_SIDE:
                nw = max(8, int(round(w / 8) * 8))
                nh = max(8, int(round(h / 8) * 8))
            else:
                scale = MAX_SIDE / longest
                nw = max(8, int(round(w * scale / 8) * 8))
                nh = max(8, int(round(h * scale / 8) * 8))
        if (nw, nh) == (w, h):
            return image
        self.console.log(f'Resize → {nw}x{nh}')
        return image.resize((nw, nh), Image.Resampling.LANCZOS)

    def resize_output(self, image):
        w, h = image.size
        longest = max(w, h)
        if longest <= OUTPUT_TARGET:
            return image
        scale = OUTPUT_TARGET / longest
        nw = max(8, int(round(w * scale)))
        nh = max(8, int(round(h * scale)))
        return image.resize((nw, nh), Image.Resampling.LANCZOS)

    def apply_mask_adjustments(self, mask, thickness):
        array = np.asarray(mask, dtype=np.uint8)
        binary = (array >= 128).astype(np.uint8)
        thickness = float(thickness)
        if thickness < 0:
            raise ValueError('mask_outline_thickness cannot be negative.')
        if thickness > 0:
            radius = int(round(thickness))
            if radius > 0:
                kernel_size = radius * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                binary = cv2.dilate(binary, kernel, iterations=1)
        return Image.fromarray((binary * 255).astype(np.uint8), 'L')

    def ensure_segmentation(self):
        if not hasattr(self, 'segmentation') or self.segmentation is None:
            self.segmentation = SegDinoSAM2()

    def load_segmentation(self):
        self.ensure_segmentation()
        self.segmentation.load_dino()
        self.segmentation.load_sam2()

    def unload_segmentation(self):
        if not hasattr(self, 'segmentation') or self.segmentation is None:
            return
        try:
            self.segmentation.unload_dino()
        except Exception:
            pass
        try:
            self.segmentation.unload_sam2()
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

    def regenerate_mask(self, image):
        if image is None or self.processing or self.model_loading or self.classification_loading:
            return
        self.segmentation_loading = True
        self.reload_mask_btn.configure(state='disabled')
        self.generate_btn.configure(state='disabled')
        self.after(0, self.show_output)
        try:
            self.console.log('Building mask...')
            mask = self.make_mask(image)
            self.mask_image = mask.copy()
            self.mask_source = 'auto'
            self.after(0, self.show_input)
            self.after(0, self.show_output)
            self.console.log('Mask ready')
        except Exception as e:
            self.mask_image = None
            self.console.log(f'Mask error: {e}')
            self.after(0, lambda err=str(e): self.console.log(f'Mask Error: {err}'))
        finally:
            self.segmentation_loading = False
            self.after(0, lambda: self.reload_mask_btn.configure(state='normal' if self.original_image is not None and not self.processing else 'disabled'))
            self.after(0, self.update_crop_button_state)
            self.after(0, self.update_generate_state)

    def reload_mask(self):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading or self.manual_segment_loading:
            return
        if self.original_image is None:
            return
        if self.manual_segment_mode:
            self.disable_manual_segment_mode()
        try:
            self.sync_config()
        except Exception as e:
            self.console.log(f'Configuration Error: {e}')
            return
        self.mask_image = None
        self.mask_source = None
        self.sd_input_image = None
        self.output_image = None
        self.output_showing_original = False
        self.generate_btn.configure(state='disabled')
        self.reload_mask_btn.configure(state='disabled')
        self.update_manual_segment_buttons()
        self.show_input()
        self.show_output()
        self.console.log('Reloading mask...')
        threading.Thread(target=self.reload_mask_worker, daemon=True).start()

    def reload_mask_worker(self):
        try:
            processed = self.preprocess_uploaded_image(self.original_image)
            self.after(0, self.show_input)
            self.regenerate_mask(processed.copy())
        except Exception as e:
            self.console.log(f'RELOAD MASK error: {e}')
            self.after(0, lambda err=str(e): self.console.log(f'Mask Reload Error: {err}'))
            self.after(0, lambda: self.reload_mask_btn.configure(state='normal' if self.original_image is not None else 'disabled'))

    def preprocess_uploaded_image(self, image):
        base = image.convert('RGB').copy()
        self.input_image = base.copy()
        cropped = self.get_cropped_image(base)
        processed = cropped.copy()
        if self.resize_var.get():
            processed = self.resize_image(processed)
        if self.esrgan_input_var.get():
            self.console.log('Enhancing...')
            processed = enhance(processed, output=False, logger=self.console.log, output_target=OUTPUT_TARGET)
            if self.resize_var.get():
                processed = self.resize_image(processed)
        self.sd_input_image = processed.copy()
        self.console.log(f'Input ready {processed.width}x{processed.height}')
        return processed

    def make_mask(self, image):
        segmentation_positive_prompt = str(self.config.get('segmentation_positive_prompt', ''))
        segmentation_negative_prompt = str(self.config.get('segmentation_negative_prompt', ''))
        thickness = float(self.config.get('mask_outline_thickness', 0.0))
        blur = float(self.config.get('mask_blur', 0.0))
        if thickness < 0:
            raise ValueError('mask_outline_thickness cannot be negative.')
        if blur < 0:
            raise ValueError('mask_blur cannot be negative.')
        self.load_segmentation()
        try:
            self.console.log('Segmenting...')
            result = self.segmentation.segment(image, segmentation_positive_prompt, segmentation_negative_prompt, thickness=0)
            masks = getattr(result, 'masks', None)
            if masks is None and isinstance(result, dict):
                masks = result.get('masks')
            if masks is None:
                mask_image = getattr(result, 'mask_image', None)
                if mask_image is None and isinstance(result, dict):
                    mask_image = result.get('mask_image')
                if mask_image is None:
                    raise RuntimeError('segdinosam2 returned no masks.')
                raw_mask = Image.fromarray(np.asarray(mask_image, dtype=np.uint8), 'L')
                count = 1
            else:
                masks = [np.asarray(mask, dtype=bool) for mask in masks if np.asarray(mask).any()]
                if not masks:
                    raise RuntimeError('segdinosam2 returned no usable masks.')
                combined = np.zeros(image.size[::-1], dtype=bool)
                for mask in masks:
                    if mask.shape != combined.shape:
                        resized = Image.fromarray((mask.astype(np.uint8) * 255), 'L').resize(image.size, Image.Resampling.NEAREST)
                        mask = np.asarray(resized, dtype=np.uint8) > 0
                    combined |= mask
                raw_mask = Image.fromarray((combined.astype(np.uint8) * 255).astype(np.uint8), 'L')
                count = len(masks)
            if raw_mask.size != image.size:
                raw_mask = raw_mask.resize(image.size, Image.Resampling.NEAREST)
            mask = self.apply_mask_adjustments(raw_mask, thickness)
            if blur > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
            self.console.log(f'Mask ready • {count} mask(s)')
            return mask.copy()
        finally:
            self.unload_segmentation()
            self.cleanup_gpu()

    def append_gender_prompts(self, positive_prompt, negative_prompt, selected_gender):
        gender_positive = {'male': 'male', 'female': 'female', 'neutral': ''}
        gender_negative = {'male': '', 'female': '', 'neutral': ''}
        positive_append = gender_positive.get(selected_gender, '')
        negative_append = gender_negative.get(selected_gender, '')
        positive_prompt = positive_prompt.strip()
        negative_prompt = negative_prompt.strip()
        if positive_append:
            positive_prompt = f'{positive_append}, {positive_prompt}' if positive_prompt else positive_append
        if negative_append:
            negative_prompt = f'{negative_append}, {negative_prompt}' if negative_prompt else negative_append
        return positive_prompt, negative_prompt

    def append_classification_prompts(self, positive_prompt, negative_prompt, classification):
        style_positive = {REAL: REAL, ANIME: ANIME, THREE_D: THREE_D, CARTOON: CARTOON}
        style_negative = {REAL: f'{ANIME}, {CARTOON}, {THREE_D}', ANIME: f'{REAL}, {THREE_D}, {CARTOON}', THREE_D: f'{REAL}, {ANIME}, {CARTOON}', CARTOON: f'{REAL}, {ANIME}, {THREE_D}'}
        positive_append = style_positive.get(classification, classification)
        negative_append = style_negative.get(classification, '')
        positive_prompt = positive_prompt.strip()
        negative_prompt = negative_prompt.strip()
        if positive_append:
            positive_prompt = f'{positive_append}, {positive_prompt}' if positive_prompt else positive_append
        if negative_append:
            negative_prompt = f'{negative_append}, {negative_prompt}' if negative_prompt else negative_append
        return positive_prompt, negative_prompt

    def has_valid_mask(self, mask):
        if mask is None:
            return False
        try:
            return np.asarray(mask, dtype=np.uint8).max() >= 10
        except Exception:
            return False

    def generate(self):
        if self.processing:
            return
        if self.model_loading or self.segmentation_loading:
            self.console.log('Models loading')
            return
        if self.classification_loading:
            self.console.log('Classification running')
            return
        if not self.models_ready:
            self.console.log('Model not ready')
            return
        if self.manual_segment_mode:
            self.disable_manual_segment_mode()
        if self.original_image is None:
            self.console.log('No image')
            return
        try:
            self.sync_config()
        except Exception as e:
            self.console.log(f'Configuration Error: {e}')
            return
        self.generation_counter += 1
        generation_id = self.generation_counter
        source = self.sd_input_image.copy() if self.sd_input_image is not None else None
        mask = self.mask_image.copy() if self.has_valid_mask(self.mask_image) else None
        if mask is None:
            self.mask_source = None
            self.console.log('No segment is present on the input preview • automatic segmentation will run')
        elif self.mask_source == 'manual':
            self.console.log('Manual segment available • skipping automatic segmentation')
        elif self.mask_source == 'auto':
            self.console.log('Automatic segment available • reusing existing mask')
        else:
            self.console.log('Existing mask available • skipping automatic segmentation')
        original_image = self.original_image.copy()
        crop_box = self.get_effective_crop_box()
        config = dict(self.config)
        classification = dict(self.classification_result or {})
        gender_result = tuple(self.gender_result) if self.gender_result is not None else ('neutral', 0.0)
        model_name = str(self.current_model_name or self.model_var.get())
        selected_class = str(self.image_class_var.get()).strip() or REAL
        selected_gender = str(self.gender_var.get()).strip().lower()
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
        self.ratio_menu.configure(state='disabled')
        self.start_time = time.time()
        self.show_output()
        self.console.log(f'Generation {generation_id} start')
        self.after(0, self.update_generate_state)
        threading.Thread(target=self.worker, args=(source, mask, original_image, crop_box, config, classification, gender_result, model_name, selected_class, selected_gender, apply_class_gender, full_image_output, esrgan_output, autosave, generation_id), daemon=True).start()

    def classify_input_image(self, file_path):
        return classify_image(file_path)

    def composite_mask(self, base, generated, mask):
        if generated.size != base.size:
            generated = generated.resize(base.size, Image.LANCZOS)
        if mask.size != base.size:
            mask = mask.resize(base.size, Image.Resampling.NEAREST)
        return Image.composite(generated, base, mask).convert('RGB')

    def worker(self, source, mask, original_image, crop_box, config, classification, gender_result, model_name, selected_class, selected_gender, apply_class_gender, full_image_output, esrgan_output, autosave, generation_id):
        try:
            if selected_gender not in ('male', 'female', 'neutral'):
                selected_gender = 'neutral'
            detected_gender = str(gender_result[0]).strip().lower()
            if detected_gender not in ('male', 'female', 'neutral'):
                detected_gender = 'neutral'
            steps = int(config.get('steps'))
            cfg = float(config.get('cfg'))
            strength = float(config.get('strength'))
            seed = int(config.get('seed', -1))
            guidance_rescale = float(config.get('guidance_rescale'))
            positive_prompt = str(config.get('positive_prompt', ''))
            negative_prompt = str(config.get('negative_prompt', ''))
            if apply_class_gender:
                positive_prompt, negative_prompt = self.append_gender_prompts(positive_prompt, negative_prompt, selected_gender)
                positive_prompt, negative_prompt = self.append_classification_prompts(positive_prompt, negative_prompt, selected_class)
            if source is None:
                self.console.log('Preprocessing...')
                source = self.preprocess_uploaded_image(original_image)
            else:
                source = source.copy()
            if not self.has_valid_mask(mask):
                self.segmentation_loading = True
                self.after(0, self.update_crop_button_state)
                self.console.log('No segment is present on the input preview • running automatic segmentation...')
                mask = self.make_mask(source)
                self.mask_image = mask.copy()
                self.mask_source = 'auto'
                self.after(0, self.show_input)
                self.segmentation_loading = False
                self.after(0, self.update_crop_button_state)
            elif mask.size != source.size:
                self.console.log(f'Reusing {self.mask_source or "existing"} segment • resizing mask to processed input')
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
            self.console.log(f'Seed {seed}')
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
            self.console.log(f'Generating {w}x{h}...')
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
            with torch.inference_mode():
                result = self.pipe(prompt=positive_prompt, negative_prompt=negative_prompt, image=init, mask_image=mask, num_inference_steps=steps, guidance_scale=cfg, strength=strength, generator=generator, width=w, height=h, callback_on_step_end=progress, guidance_rescale=guidance_rescale)
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
            self.console.log(f'Generation {generation_id} done • {self.time_text(elapsed)}')
        except GenerationStopped:
            self.console.log(f'Generation {generation_id} stopped')
        except Exception as e:
            self.console.log(f'Generation {generation_id} error: {e}')
            self.after(0, lambda err=str(e): self.console.log(f'Generation Error: {err}'))
        finally:
            self.processing = False
            self.stop_requested.clear()
            self.segmentation_loading = False
            self.cleanup_gpu()
            self.offload_diffusion_model()
            self.after(0, self.update_generate_state)
            self.after(0, lambda: self.reload_mask_btn.configure(state='normal' if self.original_image is not None and not self.segmentation_loading else 'disabled'))
            self.after(0, self.update_crop_button_state)
            self.after(0, lambda: self.upload_btn.configure(state='normal'))
            self.after(0, lambda: self.model_menu.configure(state='normal'))
            self.after(0, lambda: self.image_class_menu.configure(state='normal'))
            self.after(0, lambda: self.gender_menu.configure(state='normal'))
            self.after(0, lambda: self.ratio_menu.configure(state='normal'))

    def overlay_generated_crop(self, original_image, generated_crop, crop_box):
        original = original_image.convert('RGB').copy()
        width, height = original.size
        left, top, right, bottom = crop_box
        x1 = max(0, min(width - 1, int(round(left * width))))
        y1 = max(0, min(height - 1, int(round(top * height))))
        x2 = max(x1 + 1, min(width, int(round(right * width))))
        y2 = max(y1 + 1, min(height, int(round(bottom * height))))
        target_size = (x2 - x1, y2 - y1)
        generated_crop = generated_crop.resize(target_size, Image.Resampling.LANCZOS)
        original.paste(generated_crop, (x1, y1))
        return original

    def generation_size(self, width, height):
        width = max(8, int(width))
        height = max(8, int(height))
        ratio = self.ratio_var.get().strip().upper()
        if self.resize_var.get() and ratio in RECOMMENDED_RATIO_SIZES:
            w, h = RECOMMENDED_RATIO_SIZES[ratio]
        elif self.resize_var.get():
            longest = max(width, height)
            if longest <= MAX_SIDE:
                w = max(8, int(round(width / 8) * 8))
                h = max(8, int(round(height / 8) * 8))
            else:
                scale = MAX_SIDE / longest
                w = max(8, int(round(width * scale / 8) * 8))
                h = max(8, int(round(height * scale / 8) * 8))
        else:
            w = max(64, (width // 8) * 8)
            h = max(64, (height // 8) * 8)
        self.console.log(f'Size {w}x{h} • {ratio}')
        return w, h

    def time_text(self, seconds):
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        return f'{minutes:02d}:{seconds:02d}'

    def get_effective_crop_box(self):
        if self.crop_box is None:
            return 0.0, 0.0, 1.0, 1.0
        return self.crop_box

    def get_cropped_image(self, image):
        image = image.convert('RGB')
        if self.crop_box is None:
            return image.copy()
        width, height = image.size
        left, top, right, bottom = self.crop_box
        x1 = max(0, min(width - 1, int(round(left * width))))
        y1 = max(0, min(height - 1, int(round(top * height))))
        x2 = max(x1 + 1, min(width, int(round(right * width))))
        y2 = max(y1 + 1, min(height, int(round(bottom * height))))
        return image.crop((x1, y1, x2, y2))

    def clamp_crop_box(self, box):
        left, top, right, bottom = [float(value) for value in box]
        minimum = 0.02
        left = max(0.0, min(1.0 - minimum, left))
        top = max(0.0, min(1.0 - minimum, top))
        right = max(minimum, min(1.0, right))
        bottom = max(minimum, min(1.0, bottom))
        if right - left < minimum:
            if left + minimum <= 1.0:
                right = left + minimum
            else:
                left = max(0.0, right - minimum)
        if bottom - top < minimum:
            if top + minimum <= 1.0:
                bottom = top + minimum
            else:
                top = max(0.0, bottom - minimum)
        return left, top, right, bottom

    def get_crop_ratio(self):
        value = self.ratio_var.get().strip().upper()
        left, right = value.split(':', 1)
        return float(left) / float(right)

    def clamp_fixed_crop_box(self, box, ratio):
        if self.original_image is None or ratio is None:
            return self.clamp_crop_box(box)
        left, top, right, bottom = [float(value) for value in box]
        normalized_ratio = ratio * self.original_image.height / self.original_image.width
        width = max(0.02, right - left)
        height = max(0.02, bottom - top)
        if width / height > normalized_ratio:
            height = width / normalized_ratio
        else:
            width = height * normalized_ratio
        width = min(width, 1.0, normalized_ratio)
        height = min(height, 1.0, 1.0 / normalized_ratio)
        if width < 0.02:
            width = 0.02
            height = width / normalized_ratio
        if height < 0.02:
            height = 0.02
            width = height * normalized_ratio
        if width > 1.0:
            width = 1.0
            height = width / normalized_ratio
        if height > 1.0:
            height = 1.0
            width = height * normalized_ratio
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        left = cx - width / 2.0
        right = cx + width / 2.0
        top = cy - height / 2.0
        bottom = cy + height / 2.0
        if left < 0.0:
            right -= left
            left = 0.0
        if right > 1.0:
            left -= right - 1.0
            right = 1.0
        if top < 0.0:
            bottom -= top
            top = 0.0
        if bottom > 1.0:
            top -= bottom - 1.0
            bottom = 1.0
        return self.clamp_crop_box((left, top, right, bottom))

    def mark_crop_changed(self, message):
        self.reset_preview_state()
        self.refresh_crop_related_ui(allow_reload=True)
        self.console.log(message)
        self.show_input()
        self.show_output()

    def ratio_changed(self, choice):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        if self.original_image is None:
            return
        ratio = self.get_crop_ratio()
        self.crop_box = self.clamp_fixed_crop_box(self.get_effective_crop_box(), ratio)
        self.mark_crop_changed(f'Ratio {choice}')

    def crop_handle_size(self):
        return 11

    def crop_handle_points(self, display):
        x, y, width, height = display
        left, top, right, bottom = self.get_effective_crop_box()
        x1 = x + left * width
        y1 = y + top * height
        x2 = x + right * width
        y2 = y + bottom * height
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        return {'nw': (x1, y1), 'n': (xm, y1), 'ne': (x2, y1), 'w': (x1, ym), 'e': (x2, ym), 'sw': (x1, y2), 's': (xm, y2), 'se': (x2, y2)}

    def get_crop_handle(self, event):
        if self.input_display_info is None:
            return None
        size = self.crop_handle_size() + 5
        for handle, point in self.crop_handle_points(self.input_display_info).items():
            if abs(event.x - point[0]) <= size and abs(event.y - point[1]) <= size:
                return handle
        return None

    def start_crop(self, event):
        if self.manual_segment_mode:
            self.select_manual_segment(event)
            return
        if self.original_image is None or self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        if self.input_display_info is None:
            return
        handle = self.get_crop_handle(event)
        if handle is None:
            return
        self.active_crop_handle = handle
        self.crop_box_start = self.get_effective_crop_box()
        self.cropping = True

    def update_crop_selection(self, event):
        if self.manual_segment_mode:
            return
        if not self.cropping or self.input_display_info is None or self.crop_box_start is None:
            return
        x, y, width, height = self.input_display_info
        px = max(x, min(x + width, event.x))
        py = max(y, min(y + height, event.y))
        nx = (px - x) / width
        ny = (py - y) / height
        left, top, right, bottom = self.crop_box_start
        ratio = self.get_crop_ratio()
        normalized_ratio = ratio * self.original_image.height / self.original_image.width
        if self.active_crop_handle in ('nw', 'ne', 'sw', 'se'):
            anchor_x = right if self.active_crop_handle in ('nw', 'sw') else left
            anchor_y = bottom if self.active_crop_handle in ('nw', 'ne') else top
            dx = abs(nx - anchor_x)
            dy = abs(ny - anchor_y)
            crop_width = max(0.02, dx, dy * normalized_ratio)
            crop_height = crop_width / normalized_ratio
            if self.active_crop_handle == 'nw':
                box = (anchor_x - crop_width, anchor_y - crop_height, anchor_x, anchor_y)
            elif self.active_crop_handle == 'ne':
                box = (anchor_x, anchor_y - crop_height, anchor_x + crop_width, anchor_y)
            elif self.active_crop_handle == 'sw':
                box = (anchor_x - crop_width, anchor_y, anchor_x, anchor_y + crop_height)
            else:
                box = (anchor_x, anchor_y, anchor_x + crop_width, anchor_y + crop_height)
        elif self.active_crop_handle in ('e', 'w'):
            if self.active_crop_handle == 'e':
                new_width = max(0.02, nx - left)
                center_y = (top + bottom) / 2.0
                new_height = new_width / normalized_ratio
                box = (left, center_y - new_height / 2.0, nx, center_y + new_height / 2.0)
            else:
                new_width = max(0.02, right - nx)
                center_y = (top + bottom) / 2.0
                new_height = new_width / normalized_ratio
                box = (nx, center_y - new_height / 2.0, right, center_y + new_height / 2.0)
        elif self.active_crop_handle == 's':
            new_height = max(0.02, ny - top)
            center_x = (left + right) / 2.0
            new_width = new_height * normalized_ratio
            box = (center_x - new_width / 2.0, top, center_x + new_width / 2.0, ny)
        else:
            new_height = max(0.02, bottom - ny)
            center_x = (left + right) / 2.0
            new_width = new_height * normalized_ratio
            box = (center_x - new_width / 2.0, ny, center_x + new_width / 2.0, bottom)
        self.crop_box = self.clamp_fixed_crop_box(box, ratio)
        self.draw_crop_overlay()

    def finish_crop(self, event):
        if self.manual_segment_mode:
            return
        if not self.cropping:
            return
        self.cropping = False
        self.active_crop_handle = None
        self.crop_box_start = None
        box = self.clamp_fixed_crop_box(self.crop_box if self.crop_box is not None else (0.0, 0.0, 1.0, 1.0), self.get_crop_ratio())
        if box[0] <= 0.005 and box[1] <= 0.005 and (box[2] >= 0.995 and box[3] >= 0.995):
            self.crop_box = None
        else:
            self.crop_box = box
        self.reset_preview_state()
        self.refresh_crop_related_ui(allow_reload=True)
        self.console.log('Crop changed')
        self.show_input()
        self.show_output()

    def reset_crop(self):
        if self.processing or self.model_loading or self.segmentation_loading or self.classification_loading:
            return
        if self.original_image is None:
            return
        if self.crop_box is None:
            return
        self.crop_box = self.default_crop_box_for_ratio(self.ratio_var.get())
        self.reset_preview_state()
        self.refresh_crop_related_ui(allow_reload=True)
        self.console.log('Crop reset')
        self.show_input()
        self.show_output()

    def compose_mask_for_display(self, image, fitted_size):
        if self.mask_image is None:
            return None
        display_mask = Image.new('L', image.size, 0)
        crop_box = self.get_effective_crop_box()
        width, height = image.size
        x1 = max(0, min(width - 1, int(round(crop_box[0] * width))))
        y1 = max(0, min(height - 1, int(round(crop_box[1] * height))))
        x2 = max(x1 + 1, min(width, int(round(crop_box[2] * width))))
        y2 = max(y1 + 1, min(height, int(round(crop_box[3] * height))))
        crop_size = (x2 - x1, y2 - y1)
        mask = self.mask_image.resize(crop_size, Image.Resampling.NEAREST)
        display_mask.paste(mask, (x1, y1))
        return display_mask.resize(fitted_size, Image.Resampling.NEAREST)

    def draw_crop_overlay(self):
        if self.input_display_info is None:
            return
        self.input_canvas.delete('crop_overlay')
        x, y, width, height = self.input_display_info
        left, top, right, bottom = self.get_effective_crop_box()
        x1 = x + left * width
        y1 = y + top * height
        x2 = x + right * width
        y2 = y + bottom * height
        self.input_canvas.create_rectangle(x1, y1, x2, y2, outline='#000000', width=3, dash=(8, 5), tags='crop_overlay')
        handle_size = self.crop_handle_size()
        for point in self.crop_handle_points(self.input_display_info).values():
            hx, hy = point
            self.input_canvas.create_rectangle(hx - handle_size, hy - handle_size, hx + handle_size, hy + handle_size, outline='#000000', fill='#FFFFFF', width=2, tags='crop_overlay')

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
        self.input_photo = ImageTk.PhotoImage(fitted)
        self.input_canvas.create_image(x, y, anchor='nw', image=self.input_photo)
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
        path = filedialog.asksaveasfilename(title='Save generated image', defaultextension='.png', filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg *.jpeg'), ('WebP', '*.webp')])
        if not path:
            return
        try:
            output = self.output_image
            suffix = Path(path).suffix.lower()
            if suffix in ('.jpg', '.jpeg'):
                output = output.convert('RGB')
            output.save(path)
            self.console.log('Saved')
        except Exception as e:
            self.console.log(f'Save Error: {e}')

if __name__ == '__main__':
    app = App()
    app.mainloop()