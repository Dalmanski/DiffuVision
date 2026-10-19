import os
import gc
import json
import sys
import queue
import threading
import traceback
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0,BASE_DIR)

from tkinter import filedialog,messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline,StableDiffusionInpaintPipeline,ControlNetModel,UniPCMultistepScheduler
from insightface.app import FaceAnalysis
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose.body import Keypoint
from controlnet_aux.open_pose.util import draw_bodypose
from transformers import AutoImageProcessor,AutoModelForSemanticSegmentation
from WIP.pose_editor import SkeletonEditor,JOINT_NAMES,BONES
from modules.sd_ideal import SDIdealImageMixin
CHECKPOINT_PATH = r"Z:\Models\lazymixRealAmateur_v40Inpainting.safetensors"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_openpose"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHTS = "ip-adapter-plus_sd15.safetensors"
IP_ADAPTER_FACE_WEIGHTS = "ip-adapter-plus-face_sd15.safetensors"
SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"
CONFIG_DIR = os.path.join(BASE_DIR,"config","openpose")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR,"default.json")
PIXEL_BUDGET = 512*768
TORSO_JOINTS = ("RShoulder","LShoulder","LHip","RHip")
HEAD_JOINTS = ("Nose","REye","LEye","REar","LEar")
END_JOINTS = ("RWrist","LWrist","RAnkle","LAnkle")
MODEL_MAX_SIDE = 768
DEVICE = "cuda"
LANCZOS = Image.Resampling.LANCZOS
SEP = "="*40
STRING_KEYS = ("checkpoint_path","positive_prompt","negative_prompt")
NUMBER_KEYS = ("body_steps","body_guidance_scale","identity_strength","face_identity_strength","openpose_strength","body_mask_expand","mask_feather","moved_joint_threshold","seed","face_refinement_strength","face_refinement_steps","face_refinement_cfg","face_refinement_ip_scale","face_crop_scale","face_mask_expand_x","face_mask_expand_y","face_mask_feather")
FACE_PROMPT = "high quality realistic human face, same person, same facial identity, same facial features, same eyes, same nose, same mouth, same face shape, realistic skin, natural expression, photorealistic, sharp detailed face, preserve face position and proportions"
FACE_NEGATIVE = "different person, different identity, different face, changed facial structure, shifted face, warped face, misaligned face, deformed face, distorted face, asymmetric face, bad eyes, malformed eyes, extra eyes, crossed eyes, malformed nose, malformed mouth, blurry face, low detail face, plastic skin, cartoon, anime, mutation, low quality"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

def log(*lines):
    print("\n".join(lines),flush=True)

def log_error(title):
    head = f"{'='*10} {title} {'='*10}"
    log("\n"+head)
    traceback.print_exc()
    log("="*len(head)+"\n")

def bold(size):
    return ctk.CTkFont(size=size,weight="bold")

def read_json(path):
    with open(path,"r",encoding="utf-8") as file:
        return json.load(file)

class ConsoleRedirect:
    def __init__(self,target_queue,stream_name):
        self.target_queue,self.stream_name,self.original_stream = target_queue,stream_name,getattr(sys,stream_name)

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

class PoseChangerApp(SDIdealImageMixin):
    def __init__(self,root):
        self.root = root
        root.title("AI Pose Changer - DreamShaper Inpainting + OpenPose JSON Editor + Face Refinement")
        root.minsize(1050,760)
        root.state("zoomed")
        self.source_path = self.source_image = self.result_image = self.skeleton_image = self.skeleton_pose_data = self.editor_window = None
        self.skeleton_generation_id = 0
        self.skeleton_lock = threading.Lock()
        self.config,self.config_path = {},DEFAULT_CONFIG_PATH
        self.console_queue = queue.Queue()
        sys.stdout = ConsoleRedirect(self.console_queue,"stdout")
        sys.stderr = ConsoleRedirect(self.console_queue,"stderr")
        os.makedirs(CONFIG_DIR,exist_ok=True)
        self._load_config_file(DEFAULT_CONFIG_PATH)
        self._build_ui()
        self._refresh_config_files()
        self._poll_console()
        root.after(100,self.maximize)

    def maximize(self):
        try:
            self.root.state("zoomed")
        except Exception:
            pass

    def _build_ui(self):
        self.root.grid_rowconfigure(1,weight=1)
        self.root.grid_columnconfigure(0,weight=1)
        top = ctk.CTkFrame(self.root,fg_color="transparent")
        top.grid(row=0,column=0,sticky="ew",padx=12,pady=(12,6))
        buttons = [(None,"1. Upload Source Person Image",self.load_source,{}),("btn_edit_skeleton","2. Edit Skeleton",self.open_skeleton_editor,{"state":"disabled"}),("btn_run","3. Generate New Pose",self.start_processing,{"fg_color":"#2b8cbe","hover_color":"#236f96"}),(None,"Save Comparison Image",self.save_comparison_image,{})]
        for column,(attr,text,command,options) in enumerate(buttons):
            top.grid_columnconfigure(column,weight=1)
            button = ctk.CTkButton(top,text=text,command=command,height=40,**options)
            button.grid(row=0,column=column,padx=5,pady=5,sticky="ew")
            if attr:
                setattr(self,attr,button)
        self.btn_edit_skeleton.grid_remove()
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
        ctk.CTkLabel(left,text="CONFIGURATION",font=bold(16)).grid(row=0,column=0,pady=(12,6))
        config_area = ctk.CTkScrollableFrame(left,corner_radius=8)
        config_area.grid(row=1,column=0,sticky="nsew",padx=8,pady=(4,4))
        config_area.grid_columnconfigure(0,weight=1)
        self.config_dropdown = ctk.CTkComboBox(config_area,values=["default.json"],variable=self.config_name_var,command=self.on_config_selected,height=36)
        self.config_dropdown.grid(row=0,column=0,sticky="ew",padx=6,pady=(6,4))
        self.config_text = ctk.CTkTextbox(config_area,height=450,wrap="word")
        self.config_text.grid(row=1,column=0,sticky="nsew",padx=6,pady=6)
        self._set_config_text()
        config_buttons = ctk.CTkFrame(config_area,fg_color="transparent")
        config_buttons.grid(row=2,column=0,sticky="ew",padx=6,pady=(0,6))
        for column,(text,command) in enumerate((("Apply JSON",self.apply_config_from_editor),("Save JSON",self.save_config_from_editor))):
            config_buttons.grid_columnconfigure(column,weight=1)
            ctk.CTkButton(config_buttons,text=text,command=command).grid(row=0,column=column,padx=4,pady=4,sticky="ew")
        console_frame = ctk.CTkFrame(left,border_width=1,fg_color="#000000")
        console_frame.grid(row=2,column=0,sticky="nsew",padx=8,pady=(4,8))
        console_frame.grid_rowconfigure(1,weight=1)
        console_frame.grid_columnconfigure(0,weight=1)
        console_top = ctk.CTkFrame(console_frame,fg_color="transparent")
        console_top.grid(row=0,column=0,sticky="ew",padx=6,pady=(6,2))
        console_top.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(console_top,text="CONSOLE",font=bold(14)).grid(row=0,column=0,sticky="w")
        ctk.CTkButton(console_top,text="Clear",command=self.clear_console,width=80,height=28).grid(row=0,column=1,padx=(6,0))
        self.console_text = ctk.CTkTextbox(console_frame,wrap="none",font=("Consolas",11),fg_color="#000000",text_color="#f0f0f0")
        self.console_text.grid(row=1,column=0,sticky="nsew",padx=6,pady=(2,6))
        right = ctk.CTkFrame(main,border_width=1)
        right.grid(row=0,column=1,sticky="nsew",padx=(6,0))
        right.grid_rowconfigure(1,weight=1)
        right.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(right,text="PREVIEW IMAGES",font=bold(16)).grid(row=0,column=0,pady=(12,6))
        self.preview_frame = ctk.CTkScrollableFrame(right,corner_radius=8)
        self.preview_frame.grid(row=1,column=0,sticky="nsew",padx=8,pady=(4,8))
        self.preview_frame.grid_columnconfigure(0,weight=1)
        for row,(attr,title,text,height) in enumerate((("lbl_source","SOURCE PERSON","Upload source image",320),("lbl_skeleton","OPENPOSE SKELETON","Upload a source image to generate the skeleton.",360),("lbl_result","RESULT","Generated image",450))):
            frame = ctk.CTkFrame(self.preview_frame,border_width=1)
            frame.grid(row=row,column=0,sticky="ew",padx=6,pady=6)
            frame.grid_columnconfigure(0,weight=1)
            ctk.CTkLabel(frame,text=title,font=bold(15)).grid(row=0,column=0,pady=(10,4))
            label = ctk.CTkLabel(frame,text=text,corner_radius=8,height=height)
            label.grid(row=1,column=0,sticky="ew",padx=8,pady=8)
            setattr(self,attr,label)

    def _poll_console(self):
        try:
            while True:
                self.console_text.insert("end",self.console_queue.get_nowait()[1])
                self.console_text.see("end")
        except queue.Empty:
            pass
        self.root.after(50,self._poll_console)

    def clear_console(self):
        self.console_text.delete("1.0","end")

    def _set_config_text(self):
        self.config_text.delete("1.0","end")
        self.config_text.insert("1.0",json.dumps(self.config,indent=4))

    def _config_error(self,error,title="Invalid JSON"):
        print(f"[CONFIG ERROR] {error}",flush=True)
        messagebox.showerror(title,str(error))

    def _load_config_file(self,path):
        try:
            if os.path.exists(path):
                self.config = self.validate_config(read_json(path))
            else:
                self.config = self.validate_config({})
                self._write_config_file(self.config,path)
            self.config_path = path
        except Exception:
            self.config,self.config_path = {},DEFAULT_CONFIG_PATH

    def _write_config_file(self,config,path):
        os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
        with open(path,"w",encoding="utf-8") as file:
            json.dump(config,file,indent=4)

    def _refresh_config_files(self):
        files = sorted(name for name in os.listdir(CONFIG_DIR) if name.lower().endswith(".json"))
        if "default.json" not in files:
            files.insert(0,"default.json")
        self.config_dropdown.configure(values=files)
        selected = os.path.basename(self.config_path)
        self.config_name_var.set(selected if selected in files else "default.json")

    def on_config_selected(self,selected_name):
        path = os.path.join(CONFIG_DIR,selected_name)
        try:
            self.config = self.validate_config(read_json(path))
            self.config_path = path
            self._set_config_text()
            print(f"[CONFIG] Loaded {path}.",flush=True)
        except Exception as error:
            self._config_error(error,"Config Error")

    def validate_config(self,config):
        if not isinstance(config,dict):
            raise ValueError("JSON root must be an object.")
        defaults = read_json(DEFAULT_CONFIG_PATH)
        if not isinstance(defaults,dict):
            raise ValueError("Default config JSON root must be an object.")
        m = {"checkpoint_path":CHECKPOINT_PATH,"negative_embeddings":[],"loras":[],**defaults,**config}
        if not isinstance(m["negative_embeddings"],list) or not all(isinstance(path,str) for path in m["negative_embeddings"]):
            raise ValueError('"negative_embeddings" must be a list of file paths.')
        if not isinstance(m["loras"],list) or not all(isinstance(lora,dict) and isinstance(lora.get("path"),str) and isinstance(lora.get("weight",1.0),(int,float)) and not isinstance(lora.get("weight",1.0),bool) for lora in m["loras"]):
            raise ValueError('"loras" must be a list like [{"path": "...", "weight": 0.5}].')
        for key in STRING_KEYS:
            if not isinstance(m[key],str):
                raise ValueError(f'"{key}" must be a string.')
        for key in NUMBER_KEYS:
            if isinstance(m[key],bool) or not isinstance(m[key],(int,float)):
                raise ValueError(f'"{key}" must be a number.')
        for key in ("body_steps","face_refinement_steps","seed"):
            m[key] = int(m[key])
        if m["body_steps"] < 1 or m["face_refinement_steps"] < 1:
            raise ValueError("Step values must be at least 1.")
        if any(m[k] <= 0 for k in ("body_guidance_scale","face_refinement_cfg","face_crop_scale")) or any(m[k] < 0 for k in ("identity_strength","face_identity_strength","openpose_strength","face_refinement_strength","face_refinement_ip_scale")):
            raise ValueError("Strength, CFG, and crop values contain an invalid number.")
        if any(m[k] < 0 for k in ("body_mask_expand","mask_feather","moved_joint_threshold","face_mask_expand_x","face_mask_expand_y","face_mask_feather")):
            raise ValueError("Mask values cannot be negative.")
        return m

    def _commit_editor_config(self,save=False):
        self.config = self.validate_config(json.loads(self.config_text.get("1.0","end").strip()))
        if save:
            self._write_config_file(self.config,self.config_path)
        self._set_config_text()
        return self.config

    def apply_config_from_editor(self):
        try:
            self._commit_editor_config()
            print("[CONFIG] JSON configuration applied.",flush=True)
        except Exception as error:
            self._config_error(error)

    def save_config_from_editor(self):
        try:
            self._commit_editor_config(save=True)
            self._refresh_config_files()
            print(f"[CONFIG] Saved {self.config_path}.",flush=True)
        except Exception as error:
            self._config_error(error)

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
        self.source_path,self.source_image = path,image.copy()
        self.result_image = self.skeleton_image = self.skeleton_pose_data = None
        self.show_pil_preview(self.source_image,self.lbl_source)
        with self.skeleton_lock:
            self.skeleton_generation_id += 1
            generation_id = self.skeleton_generation_id
        self.btn_edit_skeleton.configure(state="disabled")
        self.btn_edit_skeleton.grid_remove()
        self.lbl_skeleton.configure(image=None,text="Extracting OpenPose skeleton...")
        self.lbl_result.configure(image=None,text="Generated image")
        self.lbl_result.image = None
        self.status_message("Loading OpenPose and extracting skeleton...")
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
            self.root.after(0,lambda:self.display_skeleton_preview(skeleton,generation_id))
            self.root.after(0,self.show_edit_skeleton_button)
            self.root.after(0,lambda:self.status_message(f"Skeleton ready: {skeleton.width}x{skeleton.height}"))
            print(f"[SKELETON] OpenPose skeleton preview ready at {skeleton.width}x{skeleton.height}.",flush=True)
        except Exception:
            log_error("SKELETON PREVIEW ERROR")
            self.root.after(0,lambda:self.lbl_skeleton.configure(image=None,text="Could not extract OpenPose skeleton."))
            self.root.after(0,lambda:self.status_message("Skeleton extraction failed."))
        finally:
            openpose = None
            self.clear_memory()

    def prepare_openpose_image(self,image):
        image = image.convert("RGB")
        scale = MODEL_MAX_SIDE/max(image.size)
        return image.resize((max(8,int(image.width*scale)),max(8,int(image.height*scale))),LANCZOS) if scale < 1.0 else image

    def _show_preview(self,image,label,size=(700,500)):
        preview = image.copy()
        preview.thumbnail(size,LANCZOS)
        photo = ctk.CTkImage(light_image=preview,dark_image=preview,size=preview.size)
        label.configure(image=photo,text="")
        label.image = photo

    def display_skeleton_preview(self,image,generation_id):
        if generation_id == self.skeleton_generation_id:
            self._show_preview(image,self.lbl_skeleton)

    def show_pil_preview(self,image,label_widget):
        try:
            self._show_preview(image,label_widget)
        except Exception:
            log_error("IMAGE ERROR")
            messagebox.showerror("Image Error","Could not display the selected image.")

    def show_edit_skeleton_button(self):
        self.btn_edit_skeleton.grid()
        self.btn_edit_skeleton.configure(state="normal")

    def open_skeleton_editor(self):
        with self.skeleton_lock:
            ready = self.skeleton_image is not None
        if not ready or self.source_image is None:
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
            rendered = self.render_pose_json(pose_data,(int(pose_data["image_width"]),int(pose_data["image_height"])))
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
        if float(pose_data.get("image_width",0)) <= 0 or float(pose_data.get("image_height",0)) <= 0:
            raise ValueError("Pose JSON image dimensions must be positive.")
        for name,point in joints.items():
            if not isinstance(point,(list,tuple)) or len(point) < 2:
                raise ValueError(f'Joint "{name}" must contain [x,y].')
            if not np.isfinite(float(point[0])) or not np.isfinite(float(point[1])):
                raise ValueError(f'Joint "{name}" contains an invalid coordinate.')

    def pose_points(self,pose_data,size):
        width,height = int(size[0]),int(size[1])
        scale_x = width/float(pose_data.get("image_width",width))
        scale_y = height/float(pose_data.get("image_height",height))
        points = {}
        for name,point in pose_data.get("joints",{}).items():
            if name in JOINT_NAMES and isinstance(point,(list,tuple)) and len(point) >= 2:
                points[name] = (float(point[0])*scale_x,float(point[1])*scale_y)
        return points

    def render_pose_json(self,pose_data,size):
        width,height = int(size[0]),int(size[1])
        points = self.pose_points(pose_data,size)
        keypoints = [Keypoint(points[name][0]/width,points[name][1]/height) if name in points else None for name in JOINT_NAMES]
        return Image.fromarray(draw_bodypose(np.zeros((height,width,3),dtype=np.uint8),keypoints))

    def body_scale(self,points,size):
        if "RShoulder" in points and "LShoulder" in points:
            scale = float(np.hypot(points["RShoulder"][0]-points["LShoulder"][0],points["RShoulder"][1]-points["LShoulder"][1]))
        else:
            hips = [points[name] for name in ("RHip","LHip") if name in points]
            scale = float(np.hypot(points["Neck"][0]-np.mean([p[0] for p in hips]),points["Neck"][1]-np.mean([p[1] for p in hips])))*0.6 if "Neck" in points and hips else min(size)*0.15
        return min(max(scale,min(size)*0.05),min(size)*0.5)

    def draw_body_parts(self,mask,points,joints,scale,thickness_scale=1.0):
        thickness = max(8,int(scale*0.36*thickness_scale))
        pixel = lambda point:(int(round(point[0])),int(round(point[1])))
        for a,b in BONES:
            if (a in joints or b in joints) and a in points and b in points:
                cv2.line(mask,pixel(points[a]),pixel(points[b]),255,thickness)
        for name in joints:
            if name in points:
                cv2.circle(mask,pixel(points[name]),int(thickness*(0.75 if name in END_JOINTS else 0.5)),255,-1)
        if joints & set(TORSO_JOINTS):
            polygon = [pixel(points[name]) for name in TORSO_JOINTS if name in points]
            if len(polygon) >= 3:
                cv2.fillPoly(mask,[np.array(polygon,dtype=np.int32)],255)
        head = [points[name] for name in HEAD_JOINTS if name in points]
        if head and joints & set(HEAD_JOINTS+("Neck",)):
            cv2.circle(mask,pixel((np.mean([p[0] for p in head]),np.mean([p[1] for p in head]))),int(scale*0.45*thickness_scale),255,-1)
        return mask

    def detect_source_points(self,image):
        detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        poses = detector.detect_poses(np.array(image),include_hand=False,include_face=False)
        del detector
        self.clear_memory()
        if not poses:
            return {}
        best = max(poses,key=lambda pose:sum(keypoint is not None for keypoint in pose.body.keypoints))
        return {name:(keypoint.x*image.width,keypoint.y*image.height) for name,keypoint in zip(JOINT_NAMES,best.body.keypoints) if keypoint is not None}

    def segment_person(self,image):
        processor = AutoImageProcessor.from_pretrained(SEGFORMER_MODEL)
        model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL).eval()
        with torch.inference_mode():
            logits = model(**processor(images=image,return_tensors="pt")).logits
            logits = torch.nn.functional.interpolate(logits,size=(image.height,image.width),mode="bilinear",align_corners=False)
        mask = (logits.argmax(dim=1)[0].numpy() != 0).astype(np.uint8)*255
        del processor,model,logits
        self.clear_memory()
        return mask

    def changed_region(self,person_mask,source_points,new_points,size,config):
        limit = min(size)*float(config["moved_joint_threshold"])
        moved = {name for name in JOINT_NAMES if name in source_points and name in new_points and float(np.hypot(new_points[name][0]-source_points[name][0],new_points[name][1]-source_points[name][1])) > limit}
        if not moved:
            return None,moved
        empty = np.zeros((size[1],size[0]),dtype=np.uint8)
        old_core = self.draw_body_parts(empty.copy(),source_points,moved,self.body_scale(source_points,size))
        old_wide = self.draw_body_parts(empty.copy(),source_points,moved,self.body_scale(source_points,size),1.6)
        new_part = self.draw_body_parts(empty.copy(),new_points,moved,self.body_scale(new_points,size))
        return np.maximum(np.maximum(old_core,np.minimum(old_wide,person_mask)),new_part),moved

    def build_inpaint_masks(self,person_mask,pose_data,source_points,config):
        size = (person_mask.shape[1],person_mask.shape[0])
        new_points = self.pose_points(pose_data,size)
        region,moved = self.changed_region(person_mask,source_points,new_points,size,config) if source_points else (None,set())
        repaint_face = True
        if region is not None:
            repaint_face = bool(moved & set(HEAD_JOINTS+("Neck",)))
            print(f"[MASK] Moved joints: {', '.join(sorted(moved))}. Only these limbs are repainted.",flush=True)
        elif person_mask.any():
            print("[MASK] Could not compare with the source pose. Repainting the whole person.",flush=True)
            region = np.maximum(person_mask,self.draw_body_parts(np.zeros_like(person_mask),new_points,set(new_points),self.body_scale(new_points,size)))
        else:
            print("[MASK] No person found by segmentation. Regenerating the whole image.",flush=True)
            full = Image.new("L",size,255)
            return full,full,repaint_face
        expand = max(1,int(min(size)*float(config["body_mask_expand"])))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(expand*2+1,expand*2+1))
        mask = cv2.dilate(region,kernel)
        radius = max(1,int(min(size)*float(config["mask_feather"])))
        core = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*6+1,radius*6+1)))
        blend = np.minimum(cv2.GaussianBlur(core,(0,0),radius),mask)
        return Image.fromarray(mask,"L"),Image.fromarray(blend,"L"),repaint_face

    def work_size(self,image):
        scale = min(1.0,MODEL_MAX_SIDE/max(image.size),(PIXEL_BUDGET/(image.width*image.height))**0.5)
        return max(64,int(image.width*scale)//8*8),max(64,int(image.height*scale)//8*8)

    def crop_person(self,image,person_mask):
        ys,xs = np.nonzero(person_mask)
        if len(xs) == 0:
            return image
        pad = int(max(xs.max()-xs.min(),ys.max()-ys.min())*0.05)
        return image.crop((max(0,xs.min()-pad),max(0,ys.min()-pad),min(image.width,xs.max()+pad+1),min(image.height,ys.max()+pad+1)))

    def source_face_crop(self,image):
        analyzer = FaceAnalysis(name="buffalo_l",providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        analyzer.prepare(ctx_id=0,det_size=(640,640))
        face = self.detect_largest_face(image,analyzer)
        del analyzer
        self.clear_memory()
        return self.make_square_face_crop(image,face,1.5)[0] if face is not None else None

    def start_processing(self):
        if self.source_image is None:
            messagebox.showwarning("Missing Source","Please upload the source person image.")
            return
        with self.skeleton_lock:
            pose_data = json.loads(json.dumps(self.skeleton_pose_data)) if self.skeleton_pose_data is not None else None
        if pose_data is None:
            messagebox.showwarning("Skeleton Not Ready","Please open Edit Skeleton and press Apply Skeleton first.")
            return
        if not torch.cuda.is_available():
            messagebox.showerror("CUDA Error","CUDA was not detected.")
            return
        try:
            config = self._commit_editor_config(save=True)
        except Exception as error:
            self._config_error(error)
            return
        missing = [path for path in [config["checkpoint_path"]]+config["negative_embeddings"]+[lora["path"] for lora in config["loras"]] if not os.path.isfile(path)]
        if missing:
            messagebox.showerror("File Not Found","These files were not found:\n\n"+"\n".join(missing))
            return
        log(f"[GENERATION] JSON skeleton size: {pose_data['image_width']}x{pose_data['image_height']}",f"[GENERATION] JSON joints: {len(pose_data['joints'])}",f"[GENERATION] Configuration: {self.config_path}",f"[GENERATION] Checkpoint: {config['checkpoint_path']}")
        log(f"[GENERATION] Negative embeddings: {', '.join(os.path.basename(path) for path in config['negative_embeddings']) or 'none'}",f"[GENERATION] LoRAs: {', '.join(os.path.basename(lora['path'])+' @ '+str(lora.get('weight',1.0)) for lora in config['loras']) or 'none'}")
        self.btn_run.configure(state="disabled")
        self.status_message("Starting generation from JSON skeleton...")
        threading.Thread(target=self.run_ai_pipeline,args=(config,pose_data),daemon=True).start()

    def _load_pipeline(self,pipeline_class,config,**kwargs):
        pipe = pipeline_class.from_single_file(config["checkpoint_path"],torch_dtype=torch.float16,use_safetensors=True,**kwargs)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        for path in config["negative_embeddings"]:
            pipe.load_textual_inversion(path,token=self.embedding_token(path))
        if config["loras"]:
            names = []
            for index,lora in enumerate(config["loras"]):
                names.append(f"lora_{index}")
                pipe.load_lora_weights(lora["path"],adapter_name=names[-1])
            pipe.set_adapters(names,adapter_weights=[float(lora.get("weight",1.0)) for lora in config["loras"]])
        return pipe

    def embedding_token(self,path):
        return f"<{os.path.splitext(os.path.basename(path))[0]}>"

    def negative_prompt_text(self,config,text):
        return ", ".join([self.embedding_token(path) for path in config["negative_embeddings"]]+[text])

    def _attach_ip_adapter(self,pipe,weight_names,scales):
        pipe.load_ip_adapter(IP_ADAPTER_REPO,subfolder="models",weight_name=weight_names)
        pipe.set_ip_adapter_scale(scales)
        pipe.enable_model_cpu_offload()

    def run_ai_pipeline(self,config,pose_data):
        pipe = controlnet = None
        try:
            self.clear_memory()
            print("[1/9] Preparing source image...",flush=True)
            with Image.open(self.source_path) as source_file:
                source_raw = self.filter_sd_image(source_file)
            width,height = self.work_size(source_raw)
            source_img = source_raw.convert("RGB").resize((width,height),LANCZOS)
            skeleton_img = self.render_pose_json(pose_data,(width,height)).convert("RGB")
            log(f"[1/9] Source image prepared at {width}x{height}.",f"[1/9] JSON skeleton rendered at {width}x{height} for ControlNet.")
            self.status_message("Step 2/9: Segmenting person to keep the background...")
            person_mask = self.segment_person(source_img)
            source_points = self.detect_source_points(source_img)
            inpaint_mask,blend_mask,repaint_face = self.build_inpaint_masks(person_mask,pose_data,source_points,config)
            person_crop = self.crop_person(source_img,person_mask)
            print(f"[2/9] Person mask ready. Repainting {np.count_nonzero(np.asarray(inpaint_mask) > 127)*100/(width*height):.1f}% of the image, the rest is kept.",flush=True)
            self.status_message("Step 3/9: Detecting source face...")
            face_crop = self.source_face_crop(source_raw)
            print("[3/9] Source face found for the face IP-Adapter." if face_crop is not None else "[3/9] No source face found. Using the body IP-Adapter only.",flush=True)
            self.status_message("Step 4/9: Loading OpenPose ControlNet...")
            controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL,torch_dtype=torch.float16)
            self.clear_memory()
            print("[4/9] ControlNet loaded.",flush=True)
            self.status_message("Step 5/9: Loading Stable Diffusion inpainting...")
            pipe = self._load_pipeline(StableDiffusionControlNetInpaintPipeline,config,controlnet=controlnet)
            self.clear_memory()
            print(f"[5/9] Stable Diffusion inpainting loaded: {os.path.basename(config['checkpoint_path'])}",flush=True)
            self.status_message("Step 6/9: Loading IP-Adapters...")
            identity_scale,face_scale,pose_scale = float(config["identity_strength"]),float(config["face_identity_strength"]),float(config["openpose_strength"])
            weights,scales,references = [IP_ADAPTER_WEIGHTS],[identity_scale],[person_crop]
            if face_crop is not None:
                weights,scales,references = weights+[IP_ADAPTER_FACE_WEIGHTS],scales+[face_scale],references+[face_crop]
            self._attach_ip_adapter(pipe,weights,scales)
            self.clear_memory()
            log(f"[6/9] Identity strength: {identity_scale:.2f}",f"[6/9] Face identity strength: {face_scale:.2f}" if face_crop is not None else "[6/9] Face identity strength: off",f"[6/9] OpenPose strength: {pose_scale:.2f}")
            self.status_message("Step 7/9: Generating new body pose...")
            prompt,negative_prompt = self.truncate_prompt(pipe,config["positive_prompt"]),self.truncate_prompt(pipe,self.negative_prompt_text(config,config["negative_prompt"]))
            generator = torch.Generator(device=DEVICE).manual_seed(int(config["seed"]))
            with torch.inference_mode():
                body_result = pipe(prompt=prompt,negative_prompt=negative_prompt,image=source_img,mask_image=inpaint_mask,control_image=skeleton_img,ip_adapter_image=references,num_inference_steps=int(config["body_steps"]),guidance_scale=float(config["body_guidance_scale"]),controlnet_conditioning_scale=pose_scale,width=width,height=height,generator=generator).images[0].convert("RGB")
            print("[7/9] Body generation complete.",flush=True)
            pipe = controlnet = None
            self.clear_memory()
            if repaint_face:
                self.status_message("Step 8/9: Refining face only...")
                try:
                    refined_result = self.refine_face(body_result,source_raw,config)
                    print("[8/9] Face refinement complete.",flush=True)
                except Exception:
                    log("\n"+SEP,"FACE REFINEMENT ERROR",SEP)
                    traceback.print_exc()
                    log(SEP,"Keeping the original body result.")
                    refined_result = body_result.copy()
                    self.clear_memory()
            else:
                refined_result = body_result
                print("[8/9] Head did not move. Original face kept, face refinement skipped.",flush=True)
            final_result = Image.composite(refined_result.convert("RGB"),source_img,blend_mask)
            print("[9/9] Original pixels pasted back outside the repainted area.",flush=True)
            self.clear_memory()
            self.display_result(final_result)
            del body_result,refined_result,source_raw,source_img,skeleton_img,inpaint_mask,blend_mask,person_mask,person_crop,face_crop,final_result
            self.clear_memory()
            log("\n"+SEP,"GENERATION COMPLETE","Result kept in memory only.","No automatic output file was created.",SEP+"\n")
        except Exception:
            log("\n"+SEP,"GENERATION ERROR",SEP)
            traceback.print_exc()
            log(SEP+"\n")
            pipe = controlnet = None
            self.clear_memory()
            self.root.after(0,self.generation_failed)

    def truncate_prompt(self,pipe,text):
        tokens = pipe.tokenizer(text,truncation=True,max_length=pipe.tokenizer.model_max_length,return_tensors="pt").input_ids[0]
        return pipe.tokenizer.decode(tokens,skip_special_tokens=True)

    def detect_largest_face(self,image,analyzer):
        faces = analyzer.get(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))
        return max(faces,key=lambda face:(face.bbox[2]-face.bbox[0])*(face.bbox[3]-face.bbox[1])) if faces else None

    def make_square_face_crop(self,image,face,scale):
        x1,y1,x2,y2 = [int(v) for v in face.bbox]
        size = max(128,int(max(1,x2-x1,y2-y1)*float(scale)))
        left,top = (x1+x2)//2-size//2,(y1+y2)//2-size//2
        crop = Image.new("RGB",(size,size),"white")
        box = (max(0,left),max(0,top),min(image.width,left+size),min(image.height,top+size))
        if box[2] > box[0] and box[3] > box[1]:
            crop.paste(image.crop(box),(max(0,-left),max(0,-top)))
        return crop,(x1-left,y1-top,x2-left,y2-top),(left,top,left+size,top+size)

    def build_face_blend_mask(self,crop_size,face_rect,config):
        x1,y1,x2,y2 = face_rect
        expand_x,expand_y = int(max(1,x2-x1)*float(config["face_mask_expand_x"])),int(max(1,y2-y1)*float(config["face_mask_expand_y"]))
        mask = Image.new("L",(crop_size,crop_size),0)
        ImageDraw.Draw(mask).ellipse((max(0,x1-expand_x),max(0,y1-expand_y),min(crop_size,x2+expand_x),min(crop_size,y2+expand_y)),fill=255)
        return mask.filter(ImageFilter.GaussianBlur(max(1,int(crop_size*float(config["face_mask_feather"])))))

    def refine_face(self,generated_image,source_image,config):
        print("[FACE] Detecting source and generated faces...",flush=True)
        analyzer = FaceAnalysis(name="buffalo_l",providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        analyzer.prepare(ctx_id=0,det_size=(640,640))
        source_face,generated_face = self.detect_largest_face(source_image,analyzer),self.detect_largest_face(generated_image,analyzer)
        del analyzer
        self.clear_memory()
        if source_face is None or generated_face is None:
            print("[FACE] Face detection incomplete. Skipping face refinement.",flush=True)
            return generated_image
        crop_scale = float(config["face_crop_scale"])
        source_crop = self.make_square_face_crop(source_image,source_face,crop_scale)[0].resize((512,512),LANCZOS)
        generated_crop,face_rect,box = self.make_square_face_crop(generated_image,generated_face,crop_scale)
        generated_crop = generated_crop.resize((512,512),LANCZOS)
        crop_width,crop_height = max(1,box[2]-box[0]),max(1,box[3]-box[1])
        sx,sy = 512/crop_width,512/crop_height
        face_mask = self.build_face_blend_mask(512,(int(round(face_rect[0]*sx)),int(round(face_rect[1]*sy)),int(round(face_rect[2]*sx)),int(round(face_rect[3]*sy))),config)
        refinement_pipe = None
        try:
            refinement_pipe = self._load_pipeline(StableDiffusionInpaintPipeline,config)
            self._attach_ip_adapter(refinement_pipe,[IP_ADAPTER_FACE_WEIGHTS],[float(config["face_refinement_ip_scale"])])
            prompt,negative_prompt = self.truncate_prompt(refinement_pipe,FACE_PROMPT),self.truncate_prompt(refinement_pipe,self.negative_prompt_text(config,FACE_NEGATIVE))
            generator = torch.Generator(device=DEVICE).manual_seed(int(config["seed"]))
            with torch.inference_mode():
                refined_crop = refinement_pipe(prompt=prompt,negative_prompt=negative_prompt,image=generated_crop,mask_image=face_mask,ip_adapter_image=[source_crop],strength=float(config["face_refinement_strength"]),guidance_scale=float(config["face_refinement_cfg"]),num_inference_steps=int(config["face_refinement_steps"]),generator=generator).images[0].convert("RGB")
            left,top = max(0,box[0]),max(0,box[1])
            right,bottom = min(generated_image.width,box[2]),min(generated_image.height,box[3])
            if right <= left or bottom <= top:
                return generated_image
            crop_left,crop_top = left-box[0],top-box[1]
            region = tuple(int(round(v*512/d)) for v,d in zip((crop_left,crop_top,crop_left+right-left,crop_top+bottom-top),(crop_width,crop_height,crop_width,crop_height)))
            target_size = (right-left,bottom-top)
            result = generated_image.copy()
            blended = Image.composite(refined_crop.crop(region).resize(target_size,LANCZOS),result.crop((left,top,right,bottom)),face_mask.crop(region).resize(target_size,LANCZOS))
            result.paste(blended,(left,top))
            print("[FACE] Face refinement composited only inside the feathered face mask.",flush=True)
            return result
        finally:
            refinement_pipe = None
            self.clear_memory()

    def display_result(self,result):
        self.root.after(0,lambda:self.display_result_preview(result))
        self.root.after(0,lambda:self.btn_run.configure(state="normal"))

    def display_result_preview(self,result):
        self.result_image = result.convert("RGB").copy()
        self._show_preview(result,self.lbl_result,(700,650))

    def save_comparison_image(self):
        if self.source_image is None or self.skeleton_image is None or self.result_image is None:
            messagebox.showwarning("Comparison Not Ready","Please generate an output image before saving the comparison.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG Files","*.png"),("JPEG Files","*.jpg *.jpeg")])
        if not path:
            return
        try:
            images = [("Original",self.source_image),("OpenPose Skeleton",self.skeleton_image),("Output",self.result_image)]
            panel,label_height,gap,margin = 512,48,20,20
            comparison = Image.new("RGB",(margin*2+panel*len(images)+gap*(len(images)-1),margin*2+panel+label_height),"white")
            draw = ImageDraw.Draw(comparison)
            for index,(title,image) in enumerate(images):
                preview = image.convert("RGB").copy()
                preview.thumbnail((panel,panel),LANCZOS)
                x = margin+index*(panel+gap)
                comparison.paste(preview,(x+(panel-preview.width)//2,margin+(panel-preview.height)//2))
                bbox = draw.textbbox((0,0),title)
                draw.text((x+(panel-(bbox[2]-bbox[0]))//2,margin+panel+12),title,fill="black")
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
            for collect in (torch.cuda.empty_cache,torch.cuda.ipc_collect):
                try:
                    collect()
                except Exception:
                    pass

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = PoseChangerApp(root)
    root.mainloop()
