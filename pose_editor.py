import json
import math
import sys
import tkinter as tk
import traceback
from tkinter import filedialog,messagebox
import customtkinter as ctk
from PIL import Image,ImageColor,ImageDraw,ImageEnhance,ImageTk
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from controlnet_aux.open_pose import OpenposeDetector
from controlnet_aux.open_pose.body import Body
JOINT_NAMES = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar"]
BONES = [("Nose","Neck"),("Neck","RShoulder"),("RShoulder","RElbow"),("RElbow","RWrist"),("Neck","LShoulder"),("LShoulder","LElbow"),("LElbow","LWrist"),("Neck","RHip"),("RHip","RKnee"),("RKnee","RAnkle"),("Neck","LHip"),("LHip","LKnee"),("LKnee","LAnkle"),("RHip","LHip"),("REye","LEye"),("REye","REar"),("LEye","LEar")]
PARENT_MAP = {"Nose":"Neck","Neck":None,"RShoulder":"Neck","RElbow":"RShoulder","RWrist":"RElbow","LShoulder":"Neck","LElbow":"LShoulder","LWrist":"LElbow","RHip":"Neck","RKnee":"RHip","RAnkle":"RKnee","LHip":"Neck","LKnee":"LHip","LAnkle":"LKnee","REye":"Nose","LEye":"Nose","REar":"REye","LEar":"LEye"}
DEFAULT_POSE = {"Nose":(0.50,0.12),"Neck":(0.50,0.24),"RShoulder":(0.40,0.27),"RElbow":(0.33,0.40),"RWrist":(0.29,0.54),"LShoulder":(0.60,0.27),"LElbow":(0.67,0.40),"LWrist":(0.71,0.54),"RHip":(0.43,0.53),"RKnee":(0.41,0.72),"RAnkle":(0.39,0.91),"LHip":(0.57,0.53),"LKnee":(0.59,0.72),"LAnkle":(0.61,0.91),"REye":(0.46,0.10),"LEye":(0.54,0.10),"REar":(0.42,0.12),"LEar":(0.58,0.12)}
OPENPOSE_COLORS = {"Nose":"red","Neck":"orange","RShoulder":"yellow","RElbow":"green","RWrist":"cyan","LShoulder":"yellow","LElbow":"green","LWrist":"cyan","RHip":"orange","RKnee":"lime","RAnkle":"blue","LHip":"orange","LKnee":"lime","LAnkle":"blue","REye":"pink","LEye":"pink","REar":"purple","LEar":"purple"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPENPOSE_REPO = "lllyasviel/ControlNet"
OPENPOSE_BODY_FILE = "annotator/ckpts/body_pose_model.pth"
IMAGE_DARKNESS = 0.42
MIN_FIT_MARGIN = 12
MIN_VALID_KEYPOINTS = 8
DETECTION_RESOLUTION = 768
KNEE_LINE_RATIO = 0.5
MIN_ZOOM = 0.08
MAX_ZOOM = 8.0

def console_exception_handler(exc_type,exc_value,exc_traceback):
    if exc_type is KeyboardInterrupt:
        sys.__excepthook__(exc_type,exc_value,exc_traceback)
        return
    traceback.print_exception(exc_type,exc_value,exc_traceback)
sys.excepthook = console_exception_handler
class Joint:
    def __init__(self,name,x,y):
        self.name = name
        self.x = x
        self.y = y
class Bone:
    def __init__(self,a,b):
        self.a = a
        self.b = b
        self.rest_length = 0.0
class OpenPoseRigidEditor(ctk.CTkToplevel):
    def __init__(self,parent=None,original_image=None,skeleton_image=None,on_apply=None,on_close=None):
        super().__init__(parent)
        self.title("OpenPose Rigid Editor")
        self.protocol("WM_DELETE_WINDOW",self.close_editor)
        self.on_apply = on_apply
        self.on_close = on_close
        self.original_image = original_image.convert("RGB").copy() if original_image is not None else None
        self.geometry("1150x760")
        self.minsize(900,620)
        self.configure(fg_color="#161616")
        self.show_grid = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_angles = tk.BooleanVar(value=False)
        self.extendable = tk.BooleanVar(value=False)
        self.move_skeleton = tk.BooleanVar(value=False)
        self.point_visibility = {name:tk.BooleanVar(value=True) for name in JOINT_NAMES}
        self.point_checkboxes = {}
        self.joints = {}
        self.bones = []
        self.original_pose = {}
        self.detected_valid_joints = set()
        self.smart_valid_joints = set(JOINT_NAMES)
        self.inferred_joints = set()
        self.selected_joint = None
        self.dragging = False
        self.drag_mode = None
        self.drag_start_x = 0.0
        self.drag_start_y = 0.0
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.pan_last_x = 0.0
        self.pan_last_y = 0.0
        self.panning = False
        self.image = None
        self.dark_image = None
        self.image_width = 0
        self.image_height = 0
        self.fit_zoom = 1.0
        self.zoom = 1.0
        self.image_offset_x = 0.0
        self.image_offset_y = 0.0
        self.detector = None
        self.detector_body = None
        self._tk_image = None
        self.drag_reference_positions = {}
        self.move_reference_positions = {}
        self.build_rig()
        self.build_ui()
        self.bind("<Configure>",self.on_window_resize)
        self.after(100,self.initialize_window)
        self.after(100,self.maximize)
    def maximize(self):
        try:
            self.state("zoomed")
        except Exception:
            pass
    def build_rig(self):
        self.joints = {}
        for name,(x,y) in DEFAULT_POSE.items():
            self.joints[name] = Joint(name,x,y)
        self.bones = []
        for a,b in BONES:
            bone = Bone(a,b)
            bone.rest_length = self.distance(self.joints[a],self.joints[b])
            self.bones.append(bone)
        self.original_pose = {name:(joint.x,joint.y) for name,joint in self.joints.items()}
    def build_ui(self):
        self.grid_columnconfigure(0,weight=1)
        self.grid_columnconfigure(1,weight=0)
        self.grid_rowconfigure(0,weight=1)
        editor_frame = ctk.CTkFrame(self,fg_color="#111111",corner_radius=0)
        editor_frame.grid(row=0,column=0,sticky="nsew")
        editor_frame.grid_rowconfigure(0,weight=1)
        editor_frame.grid_columnconfigure(0,weight=1)
        self.canvas = tk.Canvas(editor_frame,background="#111111",highlightthickness=0)
        self.canvas.grid(row=0,column=0,sticky="nsew")
        self.canvas.bind("<Button-1>",self.on_mouse_down)
        self.canvas.bind("<B1-Motion>",self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>",self.on_mouse_up)
        self.canvas.bind("<Button-2>",self.on_middle_mouse_down)
        self.canvas.bind("<B2-Motion>",self.on_middle_mouse_drag)
        self.canvas.bind("<ButtonRelease-2>",self.on_middle_mouse_up)
        self.canvas.bind("<MouseWheel>",self.on_mousewheel)
        self.canvas.bind("<Button-4>",self.on_mousewheel)
        self.canvas.bind("<Button-5>",self.on_mousewheel)
        sidebar = ctk.CTkScrollableFrame(self,width=300,fg_color="#1b1b1b",corner_radius=0)
        sidebar.grid(row=0,column=1,sticky="nsew",padx=(8,8),pady=8)
        sidebar.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(sidebar,text="OpenPose Rigid Editor",font=ctk.CTkFont(size=20,weight="bold")).grid(row=0,column=0,sticky="ew",padx=12,pady=(12,10))
        ctk.CTkLabel(sidebar,text="Display",font=ctk.CTkFont(size=15,weight="bold")).grid(row=1,column=0,sticky="w",padx=12,pady=(6,4))
        display_frame = ctk.CTkFrame(sidebar,fg_color="transparent")
        display_frame.grid(row=2,column=0,sticky="ew",padx=8,pady=2)
        display_frame.grid_columnconfigure(0,weight=1)
        display_frame.grid_columnconfigure(1,weight=1)
        ctk.CTkCheckBox(display_frame,text="Show Grid",variable=self.show_grid,command=self.redraw).grid(row=0,column=0,sticky="w",padx=4,pady=4)
        ctk.CTkCheckBox(display_frame,text="Show Labels",variable=self.show_labels,command=self.redraw).grid(row=0,column=1,sticky="w",padx=4,pady=4)
        ctk.CTkCheckBox(display_frame,text="Show Angles",variable=self.show_angles,command=self.redraw).grid(row=1,column=0,sticky="w",padx=4,pady=4)
        ctk.CTkCheckBox(display_frame,text="Extendible Lines",variable=self.extendable,command=self.redraw).grid(row=1,column=1,sticky="w",padx=4,pady=4)
        ctk.CTkCheckBox(display_frame,text="Move Skeleton",variable=self.move_skeleton,command=self.toggle_move_skeleton).grid(row=2,column=0,columnspan=2,sticky="w",padx=4,pady=4)
        ctk.CTkLabel(sidebar,text="Show / Hide Points",font=ctk.CTkFont(size=15,weight="bold")).grid(row=3,column=0,sticky="w",padx=12,pady=(12,4))
        visibility_frame = ctk.CTkFrame(sidebar,fg_color="transparent")
        visibility_frame.grid(row=4,column=0,sticky="ew",padx=8,pady=2)
        visibility_frame.grid_columnconfigure(0,weight=1)
        visibility_frame.grid_columnconfigure(1,weight=1)
        for index,name in enumerate(JOINT_NAMES):
            row,column = divmod(index,2)
            checkbox = ctk.CTkCheckBox(visibility_frame,text=name,variable=self.point_visibility[name],command=self.redraw)
            checkbox.grid(row=row,column=column,sticky="w",padx=4,pady=2)
            self.point_checkboxes[name] = checkbox
        ctk.CTkLabel(sidebar,text="Actions",font=ctk.CTkFont(size=15,weight="bold")).grid(row=5,column=0,sticky="w",padx=12,pady=(12,4))
        actions_frame = ctk.CTkFrame(sidebar,fg_color="transparent")
        actions_frame.grid(row=6,column=0,sticky="ew",padx=8,pady=2)
        actions_frame.grid_columnconfigure(0,weight=1)
        ctk.CTkButton(actions_frame,text="Upload Image",command=self.upload_image).grid(row=0,column=0,sticky="ew",padx=4,pady=3)
        ctk.CTkButton(actions_frame,text="Load Skeleton JSON",command=self.load_skeleton_json).grid(row=1,column=0,sticky="ew",padx=4,pady=3)
        ctk.CTkButton(actions_frame,text="Save Skeleton JSON",command=self.save_skeleton_json).grid(row=2,column=0,sticky="ew",padx=4,pady=3)
        ctk.CTkButton(actions_frame,text="Reset Skeleton",command=self.reset_skeleton).grid(row=3,column=0,sticky="ew",padx=4,pady=3)
        ctk.CTkLabel(sidebar,text="Rotate",font=ctk.CTkFont(size=15,weight="bold")).grid(row=7,column=0,sticky="w",padx=12,pady=(12,4))
        rotate_frame = ctk.CTkFrame(sidebar,fg_color="transparent")
        rotate_frame.grid(row=8,column=0,sticky="ew",padx=8,pady=2)
        rotate_frame.grid_columnconfigure(0,weight=1)
        rotate_frame.grid_columnconfigure(1,weight=1)
        ctk.CTkButton(rotate_frame,text="Rotate Left",command=lambda:self.rotate_skeleton(-90)).grid(row=0,column=0,sticky="ew",padx=4,pady=3)
        ctk.CTkButton(rotate_frame,text="Rotate Right",command=lambda:self.rotate_skeleton(90)).grid(row=0,column=1,sticky="ew",padx=4,pady=3)
        ctk.CTkLabel(sidebar,text="Apply",font=ctk.CTkFont(size=15,weight="bold")).grid(row=9,column=0,sticky="w",padx=12,pady=(12,4))
        ctk.CTkButton(sidebar,text="Apply Skeleton",command=self.apply_skeleton).grid(row=10,column=0,sticky="ew",padx=12,pady=4)
        ctk.CTkLabel(sidebar,text="Controls",font=ctk.CTkFont(size=15,weight="bold")).grid(row=11,column=0,sticky="w",padx=12,pady=(12,4))
        ctk.CTkLabel(sidebar,text="Move Skeleton ON: drag anywhere on the image to move every point together. Move Skeleton OFF: left-click a point to rotate its rigid subtree, or use Extendible Lines for free point movement. Middle mouse pans. Mouse wheel zooms.",justify="left",anchor="w",wraplength=260).grid(row=12,column=0,sticky="ew",padx=12,pady=(2,16))
        self.update_point_checkbox_states()
    def initialize_window(self):
        self.center_window()
        if self.original_image is not None and self.image is None:
            self.set_editor_image(self.original_image.copy())
        self.after(50,self.fit_image)
        self.redraw()
        if self.image is not None:
            self.initialize_pose_from_original()
    def center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = max(0,(screen_width-width)//2)
        y = max(0,(screen_height-height)//2)
        self.geometry(f"{width}x{height}+{x}+{y}")
    def on_window_resize(self,event=None):
        if self.image is not None:
            self.after_idle(self.fit_image)
    def distance(self,a,b):
        return math.hypot(a.x-b.x,a.y-b.y)
    def clamp01(self,value):
        return max(0.0,min(1.0,value))
    def set_editor_image(self,image):
        self.image = image.convert("RGB")
        self.image_width,self.image_height = self.image.size
        self.dark_image = ImageEnhance.Brightness(self.image).enhance(IMAGE_DARKNESS)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.fit_image()
    def fit_image(self):
        if self.image is None:
            self.redraw()
            return
        canvas_width = max(1,self.canvas.winfo_width())
        canvas_height = max(1,self.canvas.winfo_height())
        available_width = max(1,canvas_width-(MIN_FIT_MARGIN*2))
        available_height = max(1,canvas_height-(MIN_FIT_MARGIN*2))
        self.fit_zoom = min(available_width/self.image_width,available_height/self.image_height)
        self.zoom = min(MAX_ZOOM,max(MIN_ZOOM,self.fit_zoom))
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update_offsets()
        self.redraw()
    def update_offsets(self):
        scaled_width = self.image_width*self.zoom
        scaled_height = self.image_height*self.zoom
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.image_offset_x = (canvas_width-scaled_width)/2.0
        self.image_offset_y = (canvas_height-scaled_height)/2.0
    def render_scaled_image(self):
        if self.image is None:
            return None
        self.update_offsets()
        scaled_width = max(1,int(round(self.image_width*self.zoom)))
        scaled_height = max(1,int(round(self.image_height*self.zoom)))
        base = self.dark_image if self.dark_image is not None else self.image
        return base.resize((scaled_width,scaled_height),Image.Resampling.LANCZOS)
    def screen_to_image_x(self,x):
        self.update_offsets()
        return (x-self.image_offset_x-self.pan_x)/self.zoom
    def screen_to_image_y(self,y):
        self.update_offsets()
        return (y-self.image_offset_y-self.pan_y)/self.zoom
    def image_to_screen(self,x,y):
        self.update_offsets()
        return self.image_offset_x+self.pan_x+x*self.zoom,self.image_offset_y+self.pan_y+y*self.zoom
    def normalize_to_screen(self,joint):
        return self.image_to_screen(joint.x*self.image_width,joint.y*self.image_height)
    def point_distance_screen(self,joint,x,y):
        sx,sy = self.normalize_to_screen(joint)
        return math.hypot(sx-x,sy-y)
    def find_joint_at(self,x,y):
        best = None
        best_distance = 999999
        radius = max(10,min(24,14*self.zoom/self.fit_zoom if self.fit_zoom > 0 else 14))
        for name in JOINT_NAMES:
            if not self.point_visibility[name].get():
                continue
            distance = self.point_distance_screen(self.joints[name],x,y)
            if distance <= radius and distance < best_distance:
                best = name
                best_distance = distance
        return best
    def get_descendants(self,joint_name):
        descendants = []
        pending = [joint_name]
        while pending:
            current = pending.pop(0)
            for name,parent in PARENT_MAP.items():
                if parent == current:
                    descendants.append(name)
                    pending.append(name)
        return descendants
    def get_subtree_positions(self,joint_name):
        names = [joint_name]+self.get_descendants(joint_name)
        return {name:(self.joints[name].x,self.joints[name].y) for name in names if name in self.joints}
    def rotate_subtree(self,joint_name,new_x,new_y):
        parent_name = PARENT_MAP.get(joint_name)
        if parent_name is None or parent_name not in self.joints:
            self.joints[joint_name].x = float(new_x)
            self.joints[joint_name].y = float(new_y)
            return
        if joint_name not in self.drag_reference_positions or self.image_width <= 0 or self.image_height <= 0:
            return
        parent = self.joints[parent_name]
        start_x,start_y = self.drag_reference_positions[joint_name]
        base_px = parent.x*self.image_width
        base_py = parent.y*self.image_height
        start_px = start_x*self.image_width
        start_py = start_y*self.image_height
        target_px = new_x*self.image_width
        target_py = new_y*self.image_height
        start_angle = math.atan2(start_py-base_py,start_px-base_px)
        target_angle = math.atan2(target_py-base_py,target_px-base_px)
        angle_delta = target_angle-start_angle
        distance_from_parent = math.hypot(start_px-base_px,start_py-base_py)
        if distance_from_parent <= 1e-9:
            return
        cos_a = math.cos(angle_delta)
        sin_a = math.sin(angle_delta)
        for name,(ref_x,ref_y) in self.drag_reference_positions.items():
            if name == parent_name:
                continue
            ref_px = ref_x*self.image_width
            ref_py = ref_y*self.image_height
            dx = ref_px-base_px
            dy = ref_py-base_py
            rotated_px = base_px+dx*cos_a-dy*sin_a
            rotated_py = base_py+dx*sin_a+dy*cos_a
            self.joints[name].x = rotated_px/self.image_width
            self.joints[name].y = rotated_py/self.image_height
    def on_mouse_down(self,event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        if self.move_skeleton.get():
            self.selected_joint = "__ALL__"
            self.dragging = True
            self.drag_mode = "move_skeleton"
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.move_reference_positions = {name:(self.joints[name].x,self.joints[name].y) for name in JOINT_NAMES}
            self.canvas.configure(cursor="fleur")
            self.redraw()
            return
        joint_name = self.find_joint_at(event.x,event.y)
        if joint_name is None:
            self.selected_joint = None
            self.dragging = True
            self.drag_mode = "pan_left"
            self.pan_last_x = event.x
            self.pan_last_y = event.y
            self.canvas.configure(cursor="fleur")
            self.redraw()
            return
        self.selected_joint = joint_name
        self.dragging = True
        self.drag_mode = "free" if self.extendable.get() else "rigid"
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.drag_reference_positions = self.get_subtree_positions(joint_name)
        self.canvas.configure(cursor="crosshair")
        self.redraw()
    def on_mouse_drag(self,event):
        if not self.dragging:
            return
        dx = event.x-self.last_mouse_x
        dy = event.y-self.last_mouse_y
        if self.drag_mode == "pan_left":
            self.pan_x += dx
            self.pan_y += dy
        elif self.drag_mode == "move_skeleton":
            self.move_all_joints(event.x-self.drag_start_x,event.y-self.drag_start_y)
        elif self.drag_mode == "free" and self.selected_joint in self.joints:
            image_x = self.screen_to_image_x(event.x)
            image_y = self.screen_to_image_y(event.y)
            self.joints[self.selected_joint].x = image_x/self.image_width if self.image_width else 0.5
            self.joints[self.selected_joint].y = image_y/self.image_height if self.image_height else 0.5
        elif self.drag_mode == "rigid" and self.selected_joint in self.joints:
            image_x = self.screen_to_image_x(event.x)/self.image_width if self.image_width else 0.5
            image_y = self.screen_to_image_y(event.y)/self.image_height if self.image_height else 0.5
            self.rotate_subtree(self.selected_joint,image_x,image_y)
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        self.redraw()
    def move_all_joints(self,screen_dx,screen_dy):
        if not self.move_reference_positions or self.zoom <= 0 or self.image_width <= 0 or self.image_height <= 0:
            return
        delta_x = screen_dx/(self.zoom*self.image_width)
        delta_y = screen_dy/(self.zoom*self.image_height)
        for name,(x,y) in self.move_reference_positions.items():
            self.joints[name].x = x+delta_x
            self.joints[name].y = y+delta_y
    def on_mouse_up(self,event):
        self.dragging = False
        self.drag_mode = None
        self.drag_reference_positions = {}
        self.move_reference_positions = {}
        self.canvas.configure(cursor="arrow")
    def on_middle_mouse_down(self,event):
        self.panning = True
        self.pan_last_x = event.x
        self.pan_last_y = event.y
        self.canvas.configure(cursor="fleur")
    def on_middle_mouse_drag(self,event):
        if not self.panning:
            return
        self.pan_x += event.x-self.pan_last_x
        self.pan_y += event.y-self.pan_last_y
        self.pan_last_x = event.x
        self.pan_last_y = event.y
        self.redraw()
    def on_middle_mouse_up(self,event):
        self.panning = False
        self.canvas.configure(cursor="arrow")
    def on_mousewheel(self,event):
        if self.image is None:
            return
        if hasattr(event,"delta") and event.delta != 0:
            direction = 1 if event.delta > 0 else -1
        else:
            direction = 1 if event.num == 4 else -1
        factor = 1.12 if direction > 0 else 1/1.12
        self.zoom_at_screen_point(event.x,event.y,factor)
    def zoom_at_screen_point(self,sx,sy,factor):
        old_zoom = self.zoom
        new_zoom = max(MIN_ZOOM,min(MAX_ZOOM,old_zoom*factor))
        if abs(new_zoom-old_zoom) < 1e-9:
            return
        ix = self.screen_to_image_x(sx)
        iy = self.screen_to_image_y(sy)
        self.zoom = new_zoom
        self.update_offsets()
        self.pan_x = sx-self.image_offset_x-ix*self.zoom
        self.pan_y = sy-self.image_offset_y-iy*self.zoom
        self.redraw()
    def rotate_skeleton(self,degrees):
        visible_names = [name for name in JOINT_NAMES if self.point_visibility[name].get()]
        if not visible_names or self.image_width <= 0 or self.image_height <= 0:
            return
        center_x = sum(self.joints[name].x for name in visible_names)/len(visible_names)
        center_y = sum(self.joints[name].y for name in visible_names)/len(visible_names)
        center_px = center_x*self.image_width
        center_py = center_y*self.image_height
        angle = math.radians(degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        original_positions = {name:(self.joints[name].x,self.joints[name].y) for name in visible_names}
        for name,(x,y) in original_positions.items():
            px = x*self.image_width
            py = y*self.image_height
            dx = px-center_px
            dy = py-center_py
            rotated_px = center_px+dx*cos_a-dy*sin_a
            rotated_py = center_py+dx*sin_a+dy*cos_a
            self.joints[name].x = rotated_px/self.image_width
            self.joints[name].y = rotated_py/self.image_height
        self.redraw()
    def reset_skeleton(self):
        for name,(x,y) in self.original_pose.items():
            self.joints[name].x = x
            self.joints[name].y = y
        self.selected_joint = "__ALL__" if self.move_skeleton.get() else None
        self.redraw()
    def toggle_move_skeleton(self):
        self.dragging = False
        self.drag_mode = None
        self.drag_reference_positions = {}
        self.move_reference_positions = {}
        self.selected_joint = "__ALL__" if self.move_skeleton.get() else None
        self.canvas.configure(cursor="fleur" if self.move_skeleton.get() else "arrow")
        self.redraw()
    def update_point_checkbox_states(self):
        for name,checkbox in self.point_checkboxes.items():
            checkbox.configure(state="normal")
    def draw_grid(self):
        if not self.show_grid.get() or self.image is None:
            return
        for i in range(11):
            ratio = i/10
            x0,y0 = self.image_to_screen(ratio*self.image_width,0)
            x1,y1 = self.image_to_screen(ratio*self.image_width,self.image_height)
            self.canvas.create_line(x0,y0,x1,y1,fill="#3a3a3a",width=1)
            x0,y0 = self.image_to_screen(0,ratio*self.image_height)
            x1,y1 = self.image_to_screen(self.image_width,ratio*self.image_height)
            self.canvas.create_line(x0,y0,x1,y1,fill="#3a3a3a",width=1)
    def draw_skeleton_at_scale(self):
        for a,b in BONES:
            if not self.point_visibility[a].get() or not self.point_visibility[b].get():
                continue
            ja = self.joints[a]
            jb = self.joints[b]
            x0,y0 = self.normalize_to_screen(ja)
            x1,y1 = self.normalize_to_screen(jb)
            self.canvas.create_line(x0,y0,x1,y1,fill="#ffffff",width=4,capstyle=tk.ROUND)
        for name in JOINT_NAMES:
            if not self.point_visibility[name].get():
                continue
            joint = self.joints[name]
            x,y = self.normalize_to_screen(joint)
            radius = 7 if name not in self.inferred_joints else 6
            outline = "#ffffff"
            fill = OPENPOSE_COLORS.get(name,"#ffffff")
            if self.move_skeleton.get() or name == self.selected_joint:
                outline = "#ff0000"
                radius += 2
            self.canvas.create_oval(x-radius,y-radius,x+radius,y+radius,fill=fill,outline=outline,width=2)
            if self.show_labels.get():
                self.canvas.create_text(x+10,y,text=name,anchor="w",fill="#ffffff",font=("Arial",9,"bold"))
        if self.show_angles.get():
            self.draw_angles()
    def draw_angles(self):
        angle_groups = [("RShoulder","Neck","RHip"),("LShoulder","Neck","LHip"),("Neck","RHip","RKnee"),("Neck","LHip","LKnee"),("RHip","RKnee","RAnkle"),("LHip","LKnee","LAnkle")]
        for a,b,c in angle_groups:
            if not self.point_visibility[a].get() or not self.point_visibility[b].get() or not self.point_visibility[c].get():
                continue
            p1 = self.joints[a]
            p2 = self.joints[b]
            p3 = self.joints[c]
            v1 = np.array([p1.x-p2.x,p1.y-p2.y],dtype=float)
            v2 = np.array([p3.x-p2.x,p3.y-p2.y],dtype=float)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                continue
            cos_value = float(np.dot(v1,v2)/(n1*n2))
            cos_value = max(-1.0,min(1.0,cos_value))
            angle = math.degrees(math.acos(cos_value))
            sx,sy = self.normalize_to_screen(p2)
            self.canvas.create_text(sx+15,sy+15,text=f"{angle:.1f}°",fill="#00ffcc",font=("Arial",9,"bold"))
    def redraw(self):
        if not hasattr(self,"canvas"):
            return
        self.canvas.delete("all")
        if self.image is None:
            self.canvas.create_text(max(1,self.canvas.winfo_width()/2),max(1,self.canvas.winfo_height()/2),text="Upload an image to begin",fill="#888888",font=("Arial",18))
            return
        scaled = self.render_scaled_image()
        if scaled is None:
            return
        self._tk_image = ImageTk.PhotoImage(scaled)
        x = self.image_offset_x+self.pan_x
        y = self.image_offset_y+self.pan_y
        self.canvas.create_image(x,y,image=self._tk_image,anchor="nw")
        self.draw_grid()
        self.draw_skeleton_at_scale()
    def prepare_detection_image(self,image):
        width,height = image.size
        scale = DETECTION_RESOLUTION/max(width,height)
        new_width = max(64,int(round((width*scale)/64))*64)
        new_height = max(64,int(round((height*scale)/64))*64)
        return image.resize((new_width,new_height),Image.Resampling.LANCZOS)
    def load_openpose_detector(self):
        if self.detector is not None:
            return
        body_path = hf_hub_download(repo_id=OPENPOSE_REPO,filename=OPENPOSE_BODY_FILE)
        try:
            self.detector_body = Body(body_path)
        except TypeError:
            self.detector_body = Body(body_model_path=body_path)
        if hasattr(self.detector_body,"to"):
            self.detector_body.to(DEVICE)
        try:
            self.detector = OpenposeDetector(self.detector_body)
        except TypeError:
            self.detector = OpenposeDetector(body_estimation=self.detector_body)
    def get_pose_body_keypoints(self,pose):
        if hasattr(pose,"body") and hasattr(pose.body,"keypoints"):
            return pose.body.keypoints
        if hasattr(pose,"keypoints"):
            return pose.keypoints
        if isinstance(pose,dict):
            if "body_pose" in pose:
                return pose["body_pose"]
            if "pose_keypoints_2d" in pose:
                values = pose["pose_keypoints_2d"]
                if isinstance(values,list) and len(values)%3 == 0:
                    return [values[index:index+3] for index in range(0,len(values),3)]
            return pose
        return []
    def keypoint_is_valid(self,point):
        if point is None:
            return False
        if hasattr(point,"x") and hasattr(point,"y"):
            try:
                return math.isfinite(float(point.x)) and math.isfinite(float(point.y))
            except Exception:
                return False
        if isinstance(point,(list,tuple)) and len(point) >= 2:
            try:
                x = float(point[0])
                y = float(point[1])
                score = float(point[2]) if len(point) >= 3 else 1.0
                return math.isfinite(x) and math.isfinite(y) and score > 0.05
            except Exception:
                return False
        return False
    def get_keypoint_xy(self,point):
        if hasattr(point,"x") and hasattr(point,"y"):
            return float(point.x),float(point.y)
        return float(point[0]),float(point[1])
    def keypoints_to_points(self,pose,source_width,source_height):
        points = {}
        keypoints = self.get_pose_body_keypoints(pose)
        if keypoints is None:
            return points
        if isinstance(keypoints,np.ndarray):
            keypoints = keypoints.tolist()
        for index,name in enumerate(JOINT_NAMES):
            if index >= len(keypoints):
                continue
            point = keypoints[index]
            if not self.keypoint_is_valid(point):
                continue
            x,y = self.get_keypoint_xy(point)
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                normalized_x = x
                normalized_y = y
            else:
                normalized_x = x/float(source_width)
                normalized_y = y/float(source_height)
            points[name] = (self.clamp01(normalized_x),self.clamp01(normalized_y))
        return points
    def infer_knee_on_leg_line(self,points,side):
        hip = f"{side}Hip"
        knee = f"{side}Knee"
        ankle = f"{side}Ankle"
        if knee in points or hip not in points or ankle not in points:
            return None
        hx,hy = points[hip]
        ax,ay = points[ankle]
        return self.clamp01(hx+(ax-hx)*KNEE_LINE_RATIO),self.clamp01(hy+(ay-hy)*KNEE_LINE_RATIO)
    def smart_complete_points(self,points):
        result = dict(points)
        inferred = set()
        mirror_pairs = [("RShoulder","LShoulder"),("RElbow","LElbow"),("RWrist","LWrist"),("RHip","LHip")]
        for right,left in mirror_pairs:
            if right in result and left not in result:
                rx,ry = result[right]
                result[left] = (self.clamp01(1.0-rx),ry)
                inferred.add(left)
            elif left in result and right not in result:
                lx,ly = result[left]
                result[right] = (self.clamp01(1.0-lx),ly)
                inferred.add(right)
        for side in ("R","L"):
            knee = self.infer_knee_on_leg_line(result,side)
            name = f"{side}Knee"
            if knee is not None:
                result[name] = knee
                inferred.add(name)
        return result,inferred
    def select_best_pose(self,poses,source_width,source_height):
        best_pose = None
        best_score = (-1,-1,-1)
        for pose in poses:
            points = self.keypoints_to_points(pose,source_width,source_height)
            valid_count = len(points)
            total_score = 0.0
            keypoints = self.get_pose_body_keypoints(pose)
            parts = len(keypoints)
            if hasattr(pose,"body"):
                try:
                    total_score = float(getattr(pose.body,"total_score",0.0))
                except Exception:
                    total_score = 0.0
                try:
                    parts = int(getattr(pose.body,"total_parts",parts))
                except Exception:
                    pass
            score = (valid_count,total_score,parts)
            if score > best_score:
                best_score = score
                best_pose = pose
        return best_pose
    def apply_smart_detected_pose(self,smart_points,inferred):
        self.smart_valid_joints = set(JOINT_NAMES)
        self.inferred_joints = set(inferred)
        self.detected_valid_joints = set(smart_points.keys())
        for name in JOINT_NAMES:
            if name in smart_points:
                self.joints[name].x = smart_points[name][0]
                self.joints[name].y = smart_points[name][1]
                self.point_visibility[name].set(True)
            else:
                self.joints[name].x = DEFAULT_POSE[name][0]
                self.joints[name].y = DEFAULT_POSE[name][1]
                self.point_visibility[name].set(False)
        self.original_pose = {name:(self.joints[name].x,self.joints[name].y) for name in JOINT_NAMES}
        self.update_bone_lengths_from_pose()
        self.selected_joint = None
        self.update_point_checkbox_states()
    def update_bone_lengths_from_pose(self):
        for bone in self.bones:
            if bone.a in self.joints and bone.b in self.joints:
                bone.rest_length = self.distance(self.joints[bone.a],self.joints[bone.b])
    def initialize_pose_from_original(self):
        try:
            if self.original_image is None:
                return
            self.load_openpose_detector()
            detection_image = self.prepare_detection_image(self.original_image)
            np_image = np.array(detection_image,dtype=np.uint8)
            poses = self.detector.detect_poses(np_image,include_hand=False,include_face=False)
            if not poses:
                return
            best_pose = self.select_best_pose(poses,detection_image.width,detection_image.height)
            if best_pose is None:
                return
            detected_points = self.keypoints_to_points(best_pose,detection_image.width,detection_image.height)
            if len(detected_points) < MIN_VALID_KEYPOINTS:
                return
            smart_points,inferred = self.smart_complete_points(detected_points)
            self.apply_smart_detected_pose(smart_points,inferred)
            self.redraw()
        except Exception:
            traceback.print_exc()
    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files","*.png *.jpg *.jpeg *.webp *.bmp"),("All Files","*.*")])
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
            self.load_openpose_detector()
            detection_image = self.prepare_detection_image(image)
            np_image = np.array(detection_image,dtype=np.uint8)
            poses = self.detector.detect_poses(np_image,include_hand=False,include_face=False)
            if not poses:
                raise RuntimeError("OpenPose did not detect any body pose.")
            best_pose = self.select_best_pose(poses,detection_image.width,detection_image.height)
            if best_pose is None:
                raise RuntimeError("Unable to select a detected body pose.")
            detected_points = self.keypoints_to_points(best_pose,detection_image.width,detection_image.height)
            if len(detected_points) < MIN_VALID_KEYPOINTS:
                raise RuntimeError(f"OpenPose detected only {len(detected_points)} valid body points.")
            smart_points,inferred = self.smart_complete_points(detected_points)
            self.apply_smart_detected_pose(smart_points,inferred)
            self.set_editor_image(image)
            self.update_point_checkbox_states()
            self.redraw()
        except Exception as error:
            traceback.print_exc()
            messagebox.showerror("OpenPose Detection Error",f"{error}")
    def load_skeleton_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files","*.json"),("All Files","*.*")])
        if not path:
            return
        try:
            with open(path,"r",encoding="utf-8") as file:
                pose_data = json.load(file)
            self.apply_pose_json(pose_data)
            self.redraw()
            messagebox.showinfo("Skeleton Loaded",f"Loaded skeleton JSON:\n{path}")
        except Exception as error:
            traceback.print_exc()
            messagebox.showerror("Skeleton JSON Error",f"Could not load skeleton JSON.\n\n{error}")
    def save_skeleton_json(self):
        try:
            pose_data = self.build_pose_json()
            if not pose_data.get("joints"):
                raise RuntimeError("No visible skeleton joints are available.")
            path = filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("JSON Files","*.json"),("All Files","*.*")])
            if not path:
                return
            with open(path,"w",encoding="utf-8") as file:
                json.dump(pose_data,file,indent=4)
            messagebox.showinfo("Skeleton Saved",f"Saved skeleton JSON:\n{path}")
        except Exception as error:
            traceback.print_exc()
            messagebox.showerror("Skeleton JSON Error",f"Could not save skeleton JSON.\n\n{error}")
    def validate_pose_json(self,pose_data):
        if not isinstance(pose_data,dict):
            raise ValueError("Skeleton JSON root must be an object.")
        joints = pose_data.get("joints")
        if not isinstance(joints,dict) or not joints:
            raise ValueError('Skeleton JSON must contain a non-empty "joints" object.')
        source_width = float(pose_data.get("image_width",0))
        source_height = float(pose_data.get("image_height",0))
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Skeleton JSON image dimensions must be positive.")
        for name,point in joints.items():
            if name not in JOINT_NAMES:
                continue
            if not isinstance(point,(list,tuple)) or len(point) < 2:
                raise ValueError(f'Joint "{name}" must contain [x,y].')
            x = float(point[0])
            y = float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f'Joint "{name}" contains an invalid coordinate.')
        return source_width,source_height
    def apply_pose_json(self,pose_data):
        source_width,source_height = self.validate_pose_json(pose_data)
        if self.image is None and self.original_image is not None:
            self.set_editor_image(self.original_image.copy())
        if self.image is None:
            raise RuntimeError("Upload an image before loading a skeleton JSON.")
        joints = pose_data.get("joints",{})
        loaded_names = set()
        self.inferred_joints = set()
        self.detected_valid_joints = set()
        self.smart_valid_joints = set()
        for name in JOINT_NAMES:
            point = joints.get(name)
            if point is not None:
                self.joints[name].x = float(point[0])/source_width
                self.joints[name].y = float(point[1])/source_height
                self.point_visibility[name].set(True)
                loaded_names.add(name)
            else:
                self.joints[name].x = DEFAULT_POSE[name][0]
                self.joints[name].y = DEFAULT_POSE[name][1]
                self.point_visibility[name].set(False)
        self.detected_valid_joints = set(loaded_names)
        self.smart_valid_joints = set(loaded_names)
        self.original_pose = {name:(self.joints[name].x,self.joints[name].y) for name in JOINT_NAMES}
        self.update_bone_lengths_from_pose()
        self.selected_joint = "__ALL__" if self.move_skeleton.get() else None
        self.update_point_checkbox_states()
    def render_skeleton_image(self):
        if self.image is None:
            return None
        width,height = self.image.size
        export = Image.new("RGB",(width,height),(0,0,0))
        draw = ImageDraw.Draw(export)
        line_width = max(3,int(min(width,height)*0.006))
        for a,b in BONES:
            if not self.point_visibility[a].get() or not self.point_visibility[b].get():
                continue
            ax,ay = self.joints[a].x*width,self.joints[a].y*height
            bx,by = self.joints[b].x*width,self.joints[b].y*height
            draw.line((ax,ay,bx,by),fill=(255,255,255),width=line_width)
        radius = max(5,int(min(width,height)*0.01))
        for name in JOINT_NAMES:
            if not self.point_visibility[name].get():
                continue
            joint = self.joints[name]
            x,y = joint.x*width,joint.y*height
            fill = ImageColor.getrgb(OPENPOSE_COLORS.get(name,"white"))
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=fill,outline=(255,255,255),width=2)
        return export
    def build_pose_json(self):
        if self.image is None:
            raise RuntimeError("No editor image is loaded.")
        joints = {}
        for name in JOINT_NAMES:
            if not self.point_visibility[name].get():
                continue
            joints[name] = [round(float(self.joints[name].x*self.image_width),4),round(float(self.joints[name].y*self.image_height),4)]
        bones = [{"a":a,"b":b} for a,b in BONES if a in joints and b in joints]
        return {"format":"OpenPose-Stickman-Rigid","version":29,"source":"pose_editor.py","image_width":int(self.image_width),"image_height":int(self.image_height),"joints":joints,"bones":bones}
    def apply_skeleton(self):
        try:
            pose_data = self.build_pose_json()
            if not pose_data.get("joints"):
                raise RuntimeError("No visible skeleton joints are available.")
            if self.on_apply is not None:
                self.on_apply(json.loads(json.dumps(pose_data)))
            self.close_editor()
        except Exception as error:
            traceback.print_exc()
            messagebox.showerror("Apply Skeleton Error",f"{error}")
    def close_editor(self):
        if self.on_close is not None:
            self.on_close()
        self.destroy()
SkeletonEditor = OpenPoseRigidEditor