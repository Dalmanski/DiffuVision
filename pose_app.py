import os
import gc
import json
import sys
import queue
import threading
import traceback
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from insightface.app import FaceAnalysis
from controlnet_aux import OpenposeDetector
from pose_editor import SkeletonEditor
CHECKPOINT_PATH = r"Z:\Comfy-Desktop\ComfyUI-Shared\models\checkpoints\DreamShaper_8_INPAINTING.inpainting.safetensors"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_openpose"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHTS = "ip-adapter-plus_sd15.safetensors"
CONFIG_DIR = r"config\openpose"
DEFAULT_CONFIG_PATH = r"config\openpose\default.json"
WIDTH = 512
HEIGHT = 512
MODEL_MAX_SIDE = 768
DEVICE = "cuda"
JOINT_NAMES = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar"]
BONES = [("Nose","Neck"),("Neck","RShoulder"),("RShoulder","RElbow"),("RElbow","RWrist"),("Neck","LShoulder"),("LShoulder","LElbow"),("LElbow","LWrist"),("Neck","RHip"),("RHip","RKnee"),("RKnee","RAnkle"),("Neck","LHip"),("LHip","LKnee"),("LKnee","LAnkle"),("RHip","LHip"),("REye","LEye"),("REye","REar"),("LEye","LEar")]
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
class ConsoleRedirect:
    def __init__(self,target_queue,stream_name):
        self.target_queue = target_queue
        self.stream_name = stream_name
        self.original_stream = getattr(sys,stream_name)
    def write(self,text):
        if text:
            self.target_queue.put((self.stream_name,text))
    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass
    def isatty(self):
        return False
class PoseChangerApp:
    def __init__(self,root):
        self.root = root
        self.root.title("AI Pose Changer - DreamShaper Inpainting + OpenPose JSON Editor + Face Refinement")
        self.root.minsize(1050,760)
        self.root.state("zoomed")
        self.source_path = None
        self.source_image = None
        self.source_preview = None
        self.skeleton_preview = None
        self.result_preview = None
        self.result_image = None
        self.original_skeleton_image = None
        self.skeleton_image = None
        self.skeleton_pose_data = None
        self.original_skeleton_pose_data = None
        self.editor_window = None
        self.skeleton_generation_id = 0
        self.skeleton_lock = threading.Lock()
        self.config = {}
        self.config_path = DEFAULT_CONFIG_PATH
        self.console_queue = queue.Queue()
        self.stdout_redirect = ConsoleRedirect(self.console_queue,"stdout")
        self.stderr_redirect = ConsoleRedirect(self.console_queue,"stderr")
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stderr_redirect
        self._ensure_config_directory()
        self._load_config_file(DEFAULT_CONFIG_PATH)
        self._build_ui()
        self._refresh_config_files()
        self._poll_console()
    def _ensure_config_directory(self):
        os.makedirs(CONFIG_DIR,exist_ok=True)
    def _build_ui(self):
        self.root.grid_rowconfigure(1,weight=1)
        self.root.grid_columnconfigure(0,weight=1)
        top = ctk.CTkFrame(self.root,fg_color="transparent")
        top.grid(row=0,column=0,sticky="ew",padx=12,pady=(12,6))
        for column in range(4):
            top.grid_columnconfigure(column,weight=1)
        ctk.CTkButton(top,text="1. Upload Source Person Image",command=self.load_source,height=40).grid(row=0,column=0,padx=5,pady=5,sticky="ew")
        self.btn_edit_skeleton = ctk.CTkButton(top,text="2. Edit Skeleton",command=self.open_skeleton_editor,height=40,state="disabled")
        self.btn_edit_skeleton.grid(row=0,column=1,padx=5,pady=5,sticky="ew")
        self.btn_edit_skeleton.grid_remove()
        self.btn_run = ctk.CTkButton(top,text="3. Generate New Pose",command=self.start_processing,height=40,fg_color="#2b8cbe",hover_color="#236f96")
        self.btn_run.grid(row=0,column=2,padx=5,pady=5,sticky="ew")
        ctk.CTkButton(top,text="Save Comparison Image",command=self.save_comparison_image,height=40).grid(row=0,column=3,padx=5,pady=5,sticky="ew")
        self.config_name_var = ctk.StringVar(value="default.json")
        main = ctk.CTkFrame(self.root)
        main.grid(row=1,column=0,sticky="nsew",padx=12,pady=8)
        main.grid_columnconfigure(0,weight=1,uniform="main")
        main.grid_columnconfigure(1,weight=2,uniform="main")
        main.grid_rowconfigure(0,weight=1)
        left = ctk.CTkFrame(main,border_width=1)
        left.grid(row=0,column=0,sticky="nsew",padx=(0,6))
        left.grid_rowconfigure(1,weight=3)
        left.grid_rowconfigure(2,weight=2)
        left.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(left,text="CONFIGURATION",font=ctk.CTkFont(size=16,weight="bold")).grid(row=0,column=0,pady=(12,6))
        config_area = ctk.CTkScrollableFrame(left,corner_radius=8)
        config_area.grid(row=1,column=0,sticky="nsew",padx=8,pady=(4,4))
        config_area.grid_columnconfigure(0,weight=1)
        self.config_dropdown = ctk.CTkComboBox(config_area,values=["default.json"],variable=self.config_name_var,command=self.on_config_selected,height=36)
        self.config_dropdown.grid(row=0,column=0,sticky="ew",padx=6,pady=(6,4))
        self.config_text = ctk.CTkTextbox(config_area,height=450,wrap="word")
        self.config_text.grid(row=1,column=0,sticky="nsew",padx=6,pady=6)
        self.config_text.insert("1.0",json.dumps(self.config,indent=4))
        config_buttons = ctk.CTkFrame(config_area,fg_color="transparent")
        config_buttons.grid(row=2,column=0,sticky="ew",padx=6,pady=(0,6))
        config_buttons.grid_columnconfigure(0,weight=1)
        config_buttons.grid_columnconfigure(1,weight=1)
        ctk.CTkButton(config_buttons,text="Apply JSON",command=self.apply_config_from_editor).grid(row=0,column=0,padx=4,pady=4,sticky="ew")
        ctk.CTkButton(config_buttons,text="Save JSON",command=self.save_config_from_editor).grid(row=0,column=1,padx=4,pady=4,sticky="ew")
        console_frame = ctk.CTkFrame(left,border_width=1,fg_color="#000000")
        console_frame.grid(row=2,column=0,sticky="nsew",padx=8,pady=(4,8))
        console_frame.grid_rowconfigure(1,weight=1)
        console_frame.grid_columnconfigure(0,weight=1)
        console_top = ctk.CTkFrame(console_frame,fg_color="transparent")
        console_top.grid(row=0,column=0,sticky="ew",padx=6,pady=(6,2))
        console_top.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(console_top,text="CONSOLE",font=ctk.CTkFont(size=14,weight="bold")).grid(row=0,column=0,sticky="w")
        ctk.CTkButton(console_top,text="Clear",command=self.clear_console,width=80,height=28).grid(row=0,column=1,padx=(6,0))
        self.console_text = ctk.CTkTextbox(console_frame,wrap="none",font=("Consolas",11),fg_color="#000000",text_color="#f0f0f0")
        self.console_text.grid(row=1,column=0,sticky="nsew",padx=6,pady=(2,6))
        right = ctk.CTkFrame(main,border_width=1)
        right.grid(row=0,column=1,sticky="nsew",padx=(6,0))
        right.grid_rowconfigure(1,weight=1)
        right.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(right,text="PREVIEW IMAGES",font=ctk.CTkFont(size=16,weight="bold")).grid(row=0,column=0,pady=(12,6))
        self.preview_frame = ctk.CTkScrollableFrame(right,corner_radius=8)
        self.preview_frame.grid(row=1,column=0,sticky="nsew",padx=8,pady=(4,8))
        self.preview_frame.grid_columnconfigure(0,weight=1)
        source_frame = ctk.CTkFrame(self.preview_frame,border_width=1)
        source_frame.grid(row=0,column=0,sticky="ew",padx=6,pady=6)
        source_frame.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(source_frame,text="SOURCE PERSON",font=ctk.CTkFont(size=15,weight="bold")).grid(row=0,column=0,pady=(10,4))
        self.lbl_source = ctk.CTkLabel(source_frame,text="Upload source image",corner_radius=8,height=320)
        self.lbl_source.grid(row=1,column=0,sticky="ew",padx=8,pady=8)
        pose_frame = ctk.CTkFrame(self.preview_frame,border_width=1)
        pose_frame.grid(row=1,column=0,sticky="ew",padx=6,pady=6)
        pose_frame.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(pose_frame,text="OPENPOSE SKELETON",font=ctk.CTkFont(size=15,weight="bold")).grid(row=0,column=0,pady=(10,4))
        self.lbl_skeleton = ctk.CTkLabel(pose_frame,text="Upload a source image to generate the skeleton.",corner_radius=8,height=360)
        self.lbl_skeleton.grid(row=1,column=0,sticky="ew",padx=8,pady=8)
        result_frame = ctk.CTkFrame(self.preview_frame,border_width=1)
        result_frame.grid(row=2,column=0,sticky="ew",padx=6,pady=6)
        result_frame.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(result_frame,text="RESULT",font=ctk.CTkFont(size=15,weight="bold")).grid(row=0,column=0,pady=(10,4))
        self.lbl_result = ctk.CTkLabel(result_frame,text="Generated image",corner_radius=8,height=450)
        self.lbl_result.grid(row=1,column=0,sticky="ew",padx=8,pady=8)
    def _poll_console(self):
        try:
            while True:
                _,text = self.console_queue.get_nowait()
                self.console_text.insert("end",text)
                self.console_text.see("end")
        except queue.Empty:
            pass
        self.root.after(50,self._poll_console)
    def clear_console(self):
        self.console_text.delete("1.0","end")
    def _load_config_file(self,path):
        try:
            if os.path.exists(path):
                with open(path,"r",encoding="utf-8") as file:
                    self.config = self.validate_config(json.load(file))
                self.config_path = path
            else:
                self.config = self.validate_config({})
                self._write_config_file(self.config,path)
                self.config_path = path
        except Exception:
            self.config = {}
            self.config_path = DEFAULT_CONFIG_PATH
    def _write_config_file(self,config,path):
        os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
        with open(path,"w",encoding="utf-8") as file:
            json.dump(config,file,indent=4)
    def _refresh_config_files(self):
        files = sorted([name for name in os.listdir(CONFIG_DIR) if name.lower().endswith(".json")])
        if "default.json" not in files:
            files.insert(0,"default.json")
        self.config_dropdown.configure(values=files)
        selected = os.path.basename(self.config_path)
        self.config_name_var.set(selected if selected in files else "default.json")
    def on_config_selected(self,selected_name):
        path = os.path.join(CONFIG_DIR,selected_name)
        try:
            with open(path,"r",encoding="utf-8") as file:
                loaded = self.validate_config(json.load(file))
            self.config = loaded
            self.config_path = path
            self.config_text.delete("1.0","end")
            self.config_text.insert("1.0",json.dumps(self.config,indent=4))
            print(f"[CONFIG] Loaded {path}.",flush=True)
        except Exception as error:
            print(f"[CONFIG ERROR] {error}",flush=True)
            messagebox.showerror("Config Error",str(error))
    def validate_config(self,config):
        if not isinstance(config,dict):
            raise ValueError("JSON root must be an object.")
        with open(DEFAULT_CONFIG_PATH,"r",encoding="utf-8") as file:
            defaults = json.load(file)
        if not isinstance(defaults,dict):
            raise ValueError("Default config JSON root must be an object.")
        merged = defaults.copy()
        merged.update(config)
        string_keys = ["positive_prompt","negative_prompt"]
        number_keys = ["body_steps","body_guidance_scale","identity_strength","openpose_strength","seed","face_refinement_strength","face_refinement_steps","face_refinement_cfg","face_refinement_ip_scale","face_crop_scale","face_mask_expand_x","face_mask_expand_y","face_mask_feather"]
        for key in string_keys:
            if not isinstance(merged[key],str):
                raise ValueError(f'"{key}" must be a string.')
        for key in number_keys:
            if isinstance(merged[key],bool) or not isinstance(merged[key],(int,float)):
                raise ValueError(f'"{key}" must be a number.')
        merged["body_steps"] = int(merged["body_steps"])
        merged["face_refinement_steps"] = int(merged["face_refinement_steps"])
        merged["seed"] = int(merged["seed"])
        if merged["body_steps"] < 1 or merged["face_refinement_steps"] < 1:
            raise ValueError("Step values must be at least 1.")
        if merged["body_guidance_scale"] <= 0 or merged["identity_strength"] < 0 or merged["openpose_strength"] < 0 or merged["face_refinement_strength"] < 0 or merged["face_refinement_cfg"] <= 0 or merged["face_refinement_ip_scale"] < 0 or merged["face_crop_scale"] <= 0:
            raise ValueError("Strength, CFG, and crop values contain an invalid number.")
        if merged["face_mask_expand_x"] < 0 or merged["face_mask_expand_y"] < 0 or merged["face_mask_feather"] < 0:
            raise ValueError("Face mask values cannot be negative.")
        return merged
    def read_editor_config(self):
        return json.loads(self.config_text.get("1.0","end").strip())
    def apply_config_from_editor(self):
        try:
            parsed = self.validate_config(self.read_editor_config())
            self.config = parsed
            self.config_text.delete("1.0","end")
            self.config_text.insert("1.0",json.dumps(self.config,indent=4))
            print("[CONFIG] JSON configuration applied.",flush=True)
        except Exception as error:
            print(f"[CONFIG ERROR] {error}",flush=True)
            messagebox.showerror("Invalid JSON",str(error))
    def save_config_from_editor(self):
        try:
            parsed = self.validate_config(self.read_editor_config())
            self.config = parsed
            self._write_config_file(self.config,self.config_path)
            self.config_text.delete("1.0","end")
            self.config_text.insert("1.0",json.dumps(self.config,indent=4))
            self._refresh_config_files()
            print(f"[CONFIG] Saved {self.config_path}.",flush=True)
        except Exception as error:
            print(f"[CONFIG ERROR] {error}",flush=True)
            messagebox.showerror("Invalid JSON",str(error))
    def load_source(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files","*.png *.jpg *.jpeg *.webp *.bmp")])
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
            image.load()
        except Exception:
            messagebox.showerror("Image Error","Could not load the selected image.")
            return
        self.source_path = path
        self.source_image = image.copy()
        self.result_image = None
        self.original_skeleton_image = None
        self.skeleton_image = None
        self.skeleton_pose_data = None
        self.original_skeleton_pose_data = None
        self.show_pil_preview(self.source_image,self.lbl_source,"source")
        with self.skeleton_lock:
            self.skeleton_generation_id += 1
        self.btn_edit_skeleton.configure(state="disabled")
        self.btn_edit_skeleton.grid_remove()
        self.lbl_skeleton.configure(image=None,text="Extracting OpenPose skeleton...")
        self.lbl_result.configure(image=None,text="Generated image")
        self.lbl_result.image = None
        self.status_message("Loading OpenPose and extracting skeleton...")
        generation_id = self.skeleton_generation_id
        threading.Thread(target=self.prepare_skeleton_preview,args=(self.source_image.copy(),generation_id),daemon=True).start()
        print(f"[INPUT] Source image: {path}",flush=True)
    def prepare_skeleton_preview(self,image,generation_id):
        openpose = None
        try:
            preview_input = self.prepare_openpose_image(image)
            openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(DEVICE)
            skeleton = openpose(preview_input)
            if isinstance(skeleton,tuple):
                skeleton = skeleton[0]
            if not isinstance(skeleton,Image.Image):
                raise RuntimeError("OpenPose did not return a valid PIL image.")
            skeleton = skeleton.convert("RGB")
            with self.skeleton_lock:
                if generation_id != self.skeleton_generation_id:
                    return
                self.skeleton_image = skeleton.copy()
                self.original_skeleton_image = skeleton.copy()
            self.root.after(0,lambda:self.display_skeleton_preview(skeleton,generation_id))
            self.root.after(0,self.show_edit_skeleton_button)
            self.root.after(0,lambda:self.status_message(f"Skeleton ready: {skeleton.width}x{skeleton.height}"))
            print(f"[SKELETON] OpenPose skeleton preview ready at {skeleton.width}x{skeleton.height}.",flush=True)
        except Exception:
            print("\n========== SKELETON PREVIEW ERROR ==========",flush=True)
            traceback.print_exc()
            print("============================================\n",flush=True)
            self.root.after(0,lambda:self.lbl_skeleton.configure(image=None,text="Could not extract OpenPose skeleton."))
            self.root.after(0,lambda:self.status_message("Skeleton extraction failed."))
        finally:
            if openpose is not None:
                try:
                    del openpose
                except Exception:
                    pass
            self.clear_memory()
    def prepare_openpose_image(self,image):
        image = image.convert("RGB")
        scale = min(1.0,MODEL_MAX_SIDE/max(image.size))
        if scale < 1.0:
            return image.resize((max(8,int(image.width*scale)),max(8,int(image.height*scale))),Image.Resampling.LANCZOS)
        return image
    def display_skeleton_preview(self,image,generation_id):
        if generation_id != self.skeleton_generation_id:
            return
        preview = image.copy()
        preview.thumbnail((700,500),Image.Resampling.LANCZOS)
        photo = ctk.CTkImage(light_image=preview,dark_image=preview,size=preview.size)
        self.lbl_skeleton.configure(image=photo,text="")
        self.lbl_skeleton.image = photo
        self.skeleton_preview = photo
    def show_pil_preview(self,image,label_widget,image_type=None):
        try:
            preview = image.copy()
            preview.thumbnail((700,500),Image.Resampling.LANCZOS)
            photo = ctk.CTkImage(light_image=preview,dark_image=preview,size=preview.size)
            label_widget.configure(image=photo,text="")
            label_widget.image = photo
            if image_type == "source":
                self.source_preview = photo
        except Exception:
            print("\n========== IMAGE ERROR ==========",flush=True)
            traceback.print_exc()
            print("=================================\n",flush=True)
            messagebox.showerror("Image Error","Could not display the selected image.")
    def show_edit_skeleton_button(self):
        self.btn_edit_skeleton.grid()
        self.btn_edit_skeleton.configure(state="normal")
    def open_skeleton_editor(self):
        with self.skeleton_lock:
            skeleton = self.skeleton_image.copy() if self.skeleton_image is not None else None
        if skeleton is None or self.source_image is None:
            messagebox.showwarning("Skeleton Not Ready","Please wait until the skeleton preview is ready.")
            return
        self.btn_edit_skeleton.configure(state="disabled")
        self.status_message("Editing skeleton JSON...")
        self.editor_window = SkeletonEditor(self.root,self.source_image.copy(),None,self.apply_edited_skeleton,self.editor_closed)
        self.editor_window.transient(self.root)
        self.editor_window.grab_set()
        self.editor_window.focus_force()
    def apply_edited_skeleton(self,pose_data):
        try:
            if not isinstance(pose_data,dict):
                raise ValueError("The skeleton editor must return pose JSON.")
            self.validate_pose_data(pose_data)
            render_size = (int(pose_data["image_width"]),int(pose_data["image_height"]))
            rendered = self.render_pose_json(pose_data,render_size)
            with self.skeleton_lock:
                self.skeleton_pose_data = json.loads(json.dumps(pose_data))
                self.skeleton_image = rendered.copy()
            self.display_skeleton_preview(self.skeleton_image.copy(),self.skeleton_generation_id)
            self.btn_edit_skeleton.configure(state="normal")
            self.status_message(f"JSON skeleton applied: {rendered.width}x{rendered.height}")
            print(f"[SKELETON] JSON skeleton applied with {len(pose_data.get('joints',{}))} joints.",flush=True)
        except Exception as error:
            traceback.print_exc()
            messagebox.showerror("Skeleton JSON Error",str(error))
    def editor_closed(self):
        self.editor_window = None
        if self.skeleton_image is not None:
            self.show_edit_skeleton_button()
        self.status_message("Ready")
    def validate_pose_data(self,pose_data):
        if not isinstance(pose_data,dict):
            raise ValueError("Pose JSON root must be an object.")
        joints = pose_data.get("joints")
        if not isinstance(joints,dict) or not joints:
            raise ValueError('Pose JSON must contain a non-empty "joints" object.')
        width = float(pose_data.get("image_width",0))
        height = float(pose_data.get("image_height",0))
        if width <= 0 or height <= 0:
            raise ValueError("Pose JSON image dimensions must be positive.")
        for name,point in joints.items():
            if not isinstance(point,(list,tuple)) or len(point) < 2:
                raise ValueError(f'Joint "{name}" must contain [x,y].')
            if not np.isfinite(float(point[0])) or not np.isfinite(float(point[1])):
                raise ValueError(f'Joint "{name}" contains an invalid coordinate.')
    def render_pose_json(self,pose_data,size):
        source_width = float(pose_data.get("image_width",size[0]))
        source_height = float(pose_data.get("image_height",size[1]))
        joints = pose_data.get("joints",{})
        bones = pose_data.get("bones",BONES)
        width,height = int(size[0]),int(size[1])
        canvas = np.zeros((height,width,3),dtype=np.uint8)
        scale = min(width/source_width,height/source_height)
        scaled_width = source_width*scale
        scaled_height = source_height*scale
        offset_x = (width-scaled_width)/2.0
        offset_y = (height-scaled_height)/2.0
        palette = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85)]
        line_width = max(3,int(min(width,height)*0.012))
        joint_radius = max(4,int(min(width,height)*0.016))
        for index,bone in enumerate(bones):
            if not isinstance(bone,dict):
                continue
            a_name = bone.get("a")
            b_name = bone.get("b")
            if a_name not in joints or b_name not in joints:
                continue
            point_a = joints[a_name]
            point_b = joints[b_name]
            ax = int(round(float(point_a[0])*scale+offset_x))
            ay = int(round(float(point_a[1])*scale+offset_y))
            bx = int(round(float(point_b[0])*scale+offset_x))
            by = int(round(float(point_b[1])*scale+offset_y))
            cv2.line(canvas,(ax,ay),(bx,by),palette[index%len(palette)],line_width,cv2.LINE_AA)
        for index,point in enumerate(joints.values()):
            if not isinstance(point,(list,tuple)) or len(point) < 2:
                continue
            x = int(round(float(point[0])*scale+offset_x))
            y = int(round(float(point[1])*scale+offset_y))
            cv2.circle(canvas,(x,y),joint_radius,palette[index%len(palette)],-1,cv2.LINE_AA)
        return Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    def start_processing(self):
        if not self.source_image:
            messagebox.showwarning("Missing Source","Please upload the source person image.")
            return
        with self.skeleton_lock:
            pose_data = json.loads(json.dumps(self.skeleton_pose_data)) if self.skeleton_pose_data is not None else None
        if pose_data is None:
            messagebox.showwarning("Skeleton Not Ready","Please open Edit Skeleton and press Apply Skeleton first.")
            return
        if not os.path.exists(CHECKPOINT_PATH):
            messagebox.showerror("Checkpoint Not Found",f"Checkpoint not found:\n\n{CHECKPOINT_PATH}")
            return
        if not torch.cuda.is_available():
            messagebox.showerror("CUDA Error","CUDA was not detected.")
            return
        try:
            config = self.validate_config(self.read_editor_config())
            self.config = config
            self._write_config_file(config,self.config_path)
            self.config_text.delete("1.0","end")
            self.config_text.insert("1.0",json.dumps(config,indent=4))
        except Exception as error:
            print(f"[CONFIG ERROR] {error}",flush=True)
            messagebox.showerror("Invalid JSON",str(error))
            return
        print(f"[GENERATION] JSON skeleton size: {pose_data['image_width']}x{pose_data['image_height']}",flush=True)
        print(f"[GENERATION] JSON joints: {len(pose_data['joints'])}",flush=True)
        print(f"[GENERATION] Configuration: {self.config_path}",flush=True)
        self.btn_run.configure(state="disabled")
        self.status_message("Starting generation from JSON skeleton...")
        threading.Thread(target=self.run_ai_pipeline,args=(config,pose_data),daemon=True).start()
    def get_model_size(self,image_size):
        return WIDTH,HEIGHT
    def prepare_image(self,image,size):
        image = image.convert("RGB")
        ratio = min(size[0]/image.width,size[1]/image.height)
        new_width = max(1,int(image.width*ratio))
        new_height = max(1,int(image.height*ratio))
        image = image.resize((new_width,new_height),Image.Resampling.LANCZOS)
        canvas = Image.new("RGB",size,"white")
        x = (size[0]-new_width)//2
        y = (size[1]-new_height)//2
        canvas.paste(image,(x,y))
        return canvas
    def run_ai_pipeline(self,config,pose_data):
        pipe = None
        controlnet = None
        source_img = None
        skeleton_img = None
        body_result = None
        inpaint_mask = None
        try:
            self.clear_memory()
            print("[1/8] Preparing source image...",flush=True)
            source_raw = Image.open(self.source_path).convert("RGB")
            source_img = self.prepare_image(source_raw,(WIDTH,HEIGHT))
            del source_raw
            skeleton_img = self.render_pose_json(pose_data,(WIDTH,HEIGHT)).convert("RGB")
            inpaint_mask = Image.new("L",(WIDTH,HEIGHT),255)
            self.clear_memory()
            print("[1/8] Source image prepared at 512x512.",flush=True)
            print("[2/8] JSON skeleton rendered at 512x512 for ControlNet.",flush=True)
            self.status_message("Step 3/8: Loading OpenPose ControlNet...")
            controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL,torch_dtype=torch.float16)
            self.clear_memory()
            print("[3/8] ControlNet loaded.",flush=True)
            self.status_message("Step 4/8: Loading DreamShaper inpainting...")
            pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(CHECKPOINT_PATH,controlnet=controlnet,torch_dtype=torch.float16,use_safetensors=True)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            self.clear_memory()
            print("[4/8] Stable Diffusion inpainting loaded.",flush=True)
            self.status_message("Step 5/8: Loading IP-Adapter Plus...")
            pipe.load_ip_adapter(IP_ADAPTER_REPO,subfolder="models",weight_name=IP_ADAPTER_WEIGHTS)
            identity_scale = float(config["identity_strength"])
            pose_scale = float(config["openpose_strength"])
            pipe.set_ip_adapter_scale(identity_scale)
            pipe.enable_model_cpu_offload()
            self.clear_memory()
            print(f"[5/8] Identity strength: {identity_scale:.2f}",flush=True)
            print(f"[5/8] OpenPose strength: {pose_scale:.2f}",flush=True)
            self.status_message("Step 6/8: Generating body from JSON skeleton...")
            prompt = self.truncate_prompt(pipe,config["positive_prompt"])
            negative_prompt = self.truncate_prompt(pipe,config["negative_prompt"])
            generator = torch.Generator(device=DEVICE).manual_seed(int(config["seed"]))
            with torch.inference_mode():
                body_result = pipe(prompt=prompt,negative_prompt=negative_prompt,image=source_img,mask_image=inpaint_mask,control_image=skeleton_img,ip_adapter_image=source_img,num_inference_steps=int(config["body_steps"]),guidance_scale=float(config["body_guidance_scale"]),controlnet_conditioning_scale=pose_scale,width=WIDTH,height=HEIGHT,generator=generator).images[0]
            body_result = body_result.convert("RGB")
            print("[6/8] Body generation complete.",flush=True)
            del pipe
            pipe = None
            del controlnet
            controlnet = None
            self.clear_memory()
            self.status_message("Step 7/8: Refining face only...")
            try:
                final_result = self.refine_face(body_result,source_img,config)
                print("[7/8] Face refinement complete.",flush=True)
            except Exception:
                print("\n========================================",flush=True)
                print("FACE REFINEMENT ERROR",flush=True)
                print("========================================",flush=True)
                traceback.print_exc()
                print("========================================",flush=True)
                print("Keeping the original body result.",flush=True)
                final_result = body_result.copy()
                self.clear_memory()
            print("[8/8] Cleaning memory...",flush=True)
            final_result = final_result.convert("RGB")
            self.clear_memory()
            self.display_result(final_result)
            del body_result
            del source_img
            del skeleton_img
            del inpaint_mask
            del final_result
            self.clear_memory()
            print("\n========================================",flush=True)
            print("GENERATION COMPLETE",flush=True)
            print("Result kept in memory only.",flush=True)
            print("No automatic output file was created.",flush=True)
            print("========================================\n",flush=True)
        except Exception:
            print("\n========================================",flush=True)
            print("GENERATION ERROR",flush=True)
            print("========================================",flush=True)
            traceback.print_exc()
            print("========================================\n",flush=True)
            try:
                if pipe is not None:
                    del pipe
                if controlnet is not None:
                    del controlnet
            except Exception:
                pass
            self.clear_memory()
            self.root.after(0,self.generation_failed)
    def truncate_prompt(self,pipe,text):
        max_length = pipe.tokenizer.model_max_length
        tokens = pipe.tokenizer(text,truncation=True,max_length=max_length,return_tensors="pt").input_ids[0]
        return pipe.tokenizer.decode(tokens,skip_special_tokens=True)
    def create_face_analyzer(self):
        analyzer = FaceAnalysis(name="buffalo_l",providers=["CUDAExecutionProvider"])
        analyzer.prepare(ctx_id=0,det_size=(640,640))
        return analyzer
    def detect_largest_face(self,image,analyzer):
        image_np = np.asarray(image)
        image_bgr = cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
        faces = analyzer.get(image_bgr)
        if not faces:
            return None
        faces.sort(key=lambda face:(face.bbox[2]-face.bbox[0])*(face.bbox[3]-face.bbox[1]),reverse=True)
        return faces[0]
    def make_square_face_crop(self,image,face,scale):
        x1,y1,x2,y2 = [int(v) for v in face.bbox]
        face_width = max(1,x2-x1)
        face_height = max(1,y2-y1)
        size = max(128,int(max(face_width,face_height)*float(scale)))
        center_x = (x1+x2)//2
        center_y = (y1+y2)//2
        left = center_x-size//2
        top = center_y-size//2
        right = left+size
        bottom = top+size
        source_left = max(0,left)
        source_top = max(0,top)
        source_right = min(image.width,right)
        source_bottom = min(image.height,bottom)
        crop = Image.new("RGB",(size,size),"white")
        paste_left = max(0,-left)
        paste_top = max(0,-top)
        if source_right > source_left and source_bottom > source_top:
            region = image.crop((source_left,source_top,source_right,source_bottom))
            crop.paste(region,(paste_left,paste_top))
        face_rect = (x1-left,y1-top,x2-left,y2-top)
        return crop,face_rect,(left,top,right,bottom)
    def build_face_blend_mask(self,crop_size,face_rect,config):
        mask = Image.new("L",(crop_size,crop_size),0)
        draw = ImageDraw.Draw(mask)
        x1,y1,x2,y2 = face_rect
        width = max(1,x2-x1)
        height = max(1,y2-y1)
        expand_x = int(width*float(config["face_mask_expand_x"]))
        expand_y = int(height*float(config["face_mask_expand_y"]))
        x1 = max(0,x1-expand_x)
        y1 = max(0,y1-expand_y)
        x2 = min(crop_size,x2+expand_x)
        y2 = min(crop_size,y2+expand_y)
        draw.ellipse((x1,y1,x2,y2),fill=255)
        feather = max(1,int(crop_size*float(config["face_mask_feather"])))
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
        return mask
    def refine_face(self,generated_image,source_image,config):
        print("[FACE] Detecting source and generated faces...",flush=True)
        analyzer = self.create_face_analyzer()
        source_face = self.detect_largest_face(source_image,analyzer)
        generated_face = self.detect_largest_face(generated_image,analyzer)
        del analyzer
        self.clear_memory()
        if source_face is None or generated_face is None:
            print("[FACE] Face detection incomplete. Skipping face refinement.",flush=True)
            return generated_image
        source_crop,_,_ = self.make_square_face_crop(source_image,source_face,float(config["face_crop_scale"]))
        generated_crop,generated_face_rect,generated_box = self.make_square_face_crop(generated_image,generated_face,float(config["face_crop_scale"]))
        source_crop = source_crop.resize((512,512),Image.Resampling.LANCZOS)
        generated_crop = generated_crop.resize((512,512),Image.Resampling.LANCZOS)
        crop_width = max(1,generated_box[2]-generated_box[0])
        crop_height = max(1,generated_box[3]-generated_box[1])
        crop_scale_x = 512/crop_width
        crop_scale_y = 512/crop_height
        scaled_face_rect = (int(round(generated_face_rect[0]*crop_scale_x)),int(round(generated_face_rect[1]*crop_scale_y)),int(round(generated_face_rect[2]*crop_scale_x)),int(round(generated_face_rect[3]*crop_scale_y)))
        face_mask = self.build_face_blend_mask(512,scaled_face_rect,config)
        refinement_pipe = None
        try:
            refinement_pipe = StableDiffusionInpaintPipeline.from_single_file(CHECKPOINT_PATH,torch_dtype=torch.float16,use_safetensors=True)
            refinement_pipe.scheduler = UniPCMultistepScheduler.from_config(refinement_pipe.scheduler.config)
            refinement_pipe.load_ip_adapter(IP_ADAPTER_REPO,subfolder="models",weight_name=IP_ADAPTER_WEIGHTS)
            refinement_pipe.set_ip_adapter_scale(float(config["face_refinement_ip_scale"]))
            refinement_pipe.enable_model_cpu_offload()
            prompt = self.truncate_prompt(refinement_pipe,"high quality realistic human face, same person, same facial identity, same facial features, same eyes, same nose, same mouth, same face shape, realistic skin, natural expression, photorealistic, sharp detailed face, preserve face position and proportions")
            negative_prompt = self.truncate_prompt(refinement_pipe,"different person, different identity, different face, changed facial structure, shifted face, warped face, misaligned face, deformed face, distorted face, asymmetric face, bad eyes, malformed eyes, extra eyes, crossed eyes, malformed nose, malformed mouth, blurry face, low detail face, plastic skin, cartoon, anime, mutation, low quality")
            generator = torch.Generator(device=DEVICE).manual_seed(int(config["seed"]))
            with torch.inference_mode():
                refined_crop = refinement_pipe(prompt=prompt,negative_prompt=negative_prompt,image=generated_crop,mask_image=face_mask,ip_adapter_image=source_crop,strength=float(config["face_refinement_strength"]),guidance_scale=float(config["face_refinement_cfg"]),num_inference_steps=int(config["face_refinement_steps"]),generator=generator).images[0].convert("RGB")
            result = generated_image.copy()
            intersection_left = max(0,generated_box[0])
            intersection_top = max(0,generated_box[1])
            intersection_right = min(generated_image.width,generated_box[2])
            intersection_bottom = min(generated_image.height,generated_box[3])
            if intersection_right <= intersection_left or intersection_bottom <= intersection_top:
                return generated_image
            source_crop_left = intersection_left-generated_box[0]
            source_crop_top = intersection_top-generated_box[1]
            source_crop_right = source_crop_left+(intersection_right-intersection_left)
            source_crop_bottom = source_crop_top+(intersection_bottom-intersection_top)
            refined_region = refined_crop.crop((int(round(source_crop_left*512/max(1,crop_width))),int(round(source_crop_top*512/max(1,crop_height))),int(round(source_crop_right*512/max(1,crop_width))),int(round(source_crop_bottom*512/max(1,crop_height)))))
            mask_region = face_mask.crop((int(round(source_crop_left*512/max(1,crop_width))),int(round(source_crop_top*512/max(1,crop_height))),int(round(source_crop_right*512/max(1,crop_width))),int(round(source_crop_bottom*512/max(1,crop_height)))))
            target_size = (intersection_right-intersection_left,intersection_bottom-intersection_top)
            refined_region = refined_region.resize(target_size,Image.Resampling.LANCZOS)
            mask_region = mask_region.resize(target_size,Image.Resampling.LANCZOS)
            base_region = result.crop((intersection_left,intersection_top,intersection_right,intersection_bottom))
            blended_region = Image.composite(refined_region,base_region,mask_region)
            result.paste(blended_region,(intersection_left,intersection_top))
            print("[FACE] Face refinement composited only inside the feathered face mask.",flush=True)
            return result
        finally:
            if refinement_pipe is not None:
                try:
                    del refinement_pipe
                except Exception:
                    pass
            self.clear_memory()
    def clamp_crop_box(self,box,image_size):
        left,top,right,bottom = box
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > image_size[0]:
            left -= right-image_size[0]
            right = image_size[0]
        if bottom > image_size[1]:
            top -= bottom-image_size[1]
            bottom = image_size[1]
        left = max(0,left)
        top = max(0,top)
        right = min(image_size[0],right)
        bottom = min(image_size[1],bottom)
        if right <= left or bottom <= top:
            return (0,0,image_size[0],image_size[1])
        return (left,top,right,bottom)
    def display_result(self,result):
        self.root.after(0,lambda:self.display_result_preview(result))
        self.root.after(0,lambda:self.btn_run.configure(state="normal"))
    def display_result_preview(self,result):
        self.result_image = result.convert("RGB").copy()
        preview = result.copy()
        preview.thumbnail((700,650),Image.Resampling.LANCZOS)
        photo = ctk.CTkImage(light_image=preview,dark_image=preview,size=preview.size)
        self.lbl_result.configure(image=photo,text="")
        self.lbl_result.image = photo
        self.result_preview = photo
    def save_comparison_image(self):
        if self.source_image is None or self.skeleton_image is None or self.result_image is None:
            messagebox.showwarning("Comparison Not Ready","Please generate an output image before saving the comparison.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG Files","*.png"),("JPEG Files","*.jpg *.jpeg")])
        if not path:
            return
        try:
            images = [("Original",self.source_image.convert("RGB")),("OpenPose Skeleton",self.skeleton_image.convert("RGB")),("Output",self.result_image.convert("RGB"))]
            panel_width = 512
            panel_height = 512
            label_height = 48
            gap = 20
            margin = 20
            canvas_width = margin*2+panel_width*len(images)+gap*(len(images)-1)
            canvas_height = margin*2+panel_height+label_height
            comparison = Image.new("RGB",(canvas_width,canvas_height),"white")
            draw = ImageDraw.Draw(comparison)
            for index,(title,image) in enumerate(images):
                preview = image.copy()
                preview.thumbnail((panel_width,panel_height),Image.Resampling.LANCZOS)
                x = margin+index*(panel_width+gap)+(panel_width-preview.width)//2
                y = margin+(panel_height-preview.height)//2
                comparison.paste(preview,(x,y))
                bbox = draw.textbbox((0,0),title)
                text_width = bbox[2]-bbox[0]
                text_x = margin+index*(panel_width+gap)+(panel_width-text_width)//2
                draw.text((text_x,margin+panel_height+12),title,fill="black")
            comparison.save(path)
            print(f"[COMPARISON] Saved comparison image: {path}",flush=True)
            messagebox.showinfo("Comparison Saved",f"Saved comparison image:\n{path}")
        except Exception as error:
            traceback.print_exc()
            messagebox.showerror("Comparison Save Error",str(error))
    def status_message(self,text):
        self.root.after(0,lambda:self.root.title(f"AI Pose Changer - {text}"))
    def generation_failed(self):
        self.btn_run.configure(state="normal")
        self.status_message("Generation failed")
    def clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = PoseChangerApp(root)
    root.mainloop()