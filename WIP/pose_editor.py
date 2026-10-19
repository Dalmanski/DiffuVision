import json
import math
import sys
import traceback
import tkinter as tk
from tkinter import filedialog,messagebox
import customtkinter as ctk
from PIL import Image,ImageEnhance,ImageTk
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
ANGLE_GROUPS = [("RShoulder","Neck","RHip"),("LShoulder","Neck","LHip"),("Neck","RHip","RKnee"),("Neck","LHip","LKnee"),("RHip","RKnee","RAnkle"),("LHip","LKnee","LAnkle")]
MIRROR_PAIRS = [("RShoulder","LShoulder"),("RElbow","LElbow"),("RWrist","LWrist"),("RHip","LHip")]
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
JSON_TYPES = [("JSON Files","*.json"),("All Files","*.*")]

def console_exception_handler(exc_type,exc_value,exc_traceback):
    if exc_type is KeyboardInterrupt:
        return sys.__excepthook__(exc_type,exc_value,exc_traceback)
    traceback.print_exception(exc_type,exc_value,exc_traceback)

sys.excepthook = console_exception_handler

class PoseNotFound(RuntimeError):
    pass

class Joint:
    def __init__(self,name,x,y):
        self.name,self.x,self.y = name,x,y

class OpenPoseRigidEditor(ctk.CTkToplevel):
    def __init__(self,parent=None,original_image=None,skeleton_image=None,on_apply=None,on_close=None):
        super().__init__(parent)
        self.title("OpenPose Rigid Editor")
        self.protocol("WM_DELETE_WINDOW",self.close_editor)
        self.on_apply,self.on_close = on_apply,on_close
        self.original_image = original_image.convert("RGB").copy() if original_image is not None else None
        self.geometry("1150x760")
        self.minsize(900,620)
        self.configure(fg_color="#161616")
        self.show_grid,self.show_labels,self.show_angles,self.extendable,self.move_skeleton = (tk.BooleanVar(value=v) for v in (True,True,False,False,False))
        self.point_visibility = {name:tk.BooleanVar(value=True) for name in JOINT_NAMES}
        self.joints = {name:Joint(name,x,y) for name,(x,y) in DEFAULT_POSE.items()}
        self.original_pose = dict(DEFAULT_POSE)
        self.inferred_joints = set()
        self.selected_joint = self.drag_mode = self.image = self.dark_image = self.detector = self.detector_body = self._tk_image = None
        self.dragging = self.panning = False
        self.drag_start_x = self.drag_start_y = self.last_mouse_x = self.last_mouse_y = self.pan_x = self.pan_y = self.pan_last_x = self.pan_last_y = 0.0
        self.image_width = self.image_height = 0
        self.fit_zoom = self.zoom = 1.0
        self.image_offset_x = self.image_offset_y = 0.0
        self.drag_reference_positions,self.move_reference_positions = {},{}
        self.build_ui()
        self.bind("<Configure>",self.on_window_resize)
        self.after(100,self.initialize_window)
        self.after(100,self.maximize)

    def maximize(self):
        try:
            self.state("zoomed")
        except Exception:
            pass

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
        for sequence,handler in (("<Button-1>",self.on_mouse_down),("<B1-Motion>",self.on_mouse_drag),("<ButtonRelease-1>",self.on_mouse_up),("<Button-2>",self.on_middle_mouse_down),("<B2-Motion>",self.on_middle_mouse_drag),("<ButtonRelease-2>",self.on_middle_mouse_up),("<MouseWheel>",self.on_mousewheel),("<Button-4>",self.on_mousewheel),("<Button-5>",self.on_mousewheel)):
            self.canvas.bind(sequence,handler)
        sidebar = ctk.CTkScrollableFrame(self,width=300,fg_color="#1b1b1b",corner_radius=0)
        sidebar.grid(row=0,column=1,sticky="nsew",padx=(8,8),pady=8)
        sidebar.grid_columnconfigure(0,weight=1)
        bold = lambda size:ctk.CTkFont(size=size,weight="bold")

        def heading(row,text,pady=(12,4)):
            ctk.CTkLabel(sidebar,text=text,font=bold(15)).grid(row=row,column=0,sticky="w",padx=12,pady=pady)

        def section(row,columns=1):
            frame = ctk.CTkFrame(sidebar,fg_color="transparent")
            frame.grid(row=row,column=0,sticky="ew",padx=8,pady=2)
            for column in range(columns):
                frame.grid_columnconfigure(column,weight=1)
            return frame
        ctk.CTkLabel(sidebar,text="OpenPose Rigid Editor",font=bold(20)).grid(row=0,column=0,sticky="ew",padx=12,pady=(12,10))
        heading(1,"Display",(6,4))
        display = section(2,2)
        for index,(text,variable) in enumerate((("Show Grid",self.show_grid),("Show Labels",self.show_labels),("Show Angles",self.show_angles),("Extendible Lines",self.extendable))):
            ctk.CTkCheckBox(display,text=text,variable=variable,command=self.redraw).grid(row=index//2,column=index%2,sticky="w",padx=4,pady=4)
        ctk.CTkCheckBox(display,text="Move Skeleton",variable=self.move_skeleton,command=self.toggle_move_skeleton).grid(row=2,column=0,columnspan=2,sticky="w",padx=4,pady=4)
        heading(3,"Show / Hide Points")
        visibility = section(4,2)
        for index,name in enumerate(JOINT_NAMES):
            ctk.CTkCheckBox(visibility,text=name,variable=self.point_visibility[name],command=self.redraw).grid(row=index//2,column=index%2,sticky="w",padx=4,pady=2)
        heading(5,"Actions")
        actions = section(6)
        for row,(text,command) in enumerate((("Upload Image",self.upload_image),("Load Skeleton JSON",self.load_skeleton_json),("Save Skeleton JSON",self.save_skeleton_json),("Reset Skeleton",self.reset_skeleton))):
            ctk.CTkButton(actions,text=text,command=command).grid(row=row,column=0,sticky="ew",padx=4,pady=3)
        heading(7,"Rotate")
        rotate = section(8,2)
        for column,(text,degrees) in enumerate((("Rotate Left",-90),("Rotate Right",90))):
            ctk.CTkButton(rotate,text=text,command=lambda d=degrees:self.rotate_skeleton(d)).grid(row=0,column=column,sticky="ew",padx=4,pady=3)
        heading(9,"Apply")
        ctk.CTkButton(sidebar,text="Apply Skeleton",command=self.apply_skeleton).grid(row=10,column=0,sticky="ew",padx=12,pady=4)
        heading(11,"Controls")
        ctk.CTkLabel(sidebar,text="Move Skeleton ON: drag anywhere on the image to move every point together. Move Skeleton OFF: left-click a point to rotate its rigid subtree, or use Extendible Lines for free point movement. Middle mouse pans. Mouse wheel zooms.",justify="left",anchor="w",wraplength=260).grid(row=12,column=0,sticky="ew",padx=12,pady=(2,16))

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
        width,height = self.winfo_width(),self.winfo_height()
        self.geometry(f"{width}x{height}+{max(0,(self.winfo_screenwidth()-width)//2)}+{max(0,(self.winfo_screenheight()-height)//2)}")

    def on_window_resize(self,event=None):
        if self.image is not None:
            self.after_idle(self.fit_image)

    def clamp01(self,value):
        return max(0.0,min(1.0,value))

    def _positions(self,names):
        return {name:(self.joints[name].x,self.joints[name].y) for name in names}

    def _visible(self,*names):
        return all(self.point_visibility[name].get() for name in names)

    def set_editor_image(self,image):
        self.image = image.convert("RGB")
        self.image_width,self.image_height = self.image.size
        self.dark_image = ImageEnhance.Brightness(self.image).enhance(IMAGE_DARKNESS)
        self.pan_x = self.pan_y = 0.0
        self.fit_image()

    def fit_image(self):
        if self.image is not None:
            available_width = max(1,max(1,self.canvas.winfo_width())-MIN_FIT_MARGIN*2)
            available_height = max(1,max(1,self.canvas.winfo_height())-MIN_FIT_MARGIN*2)
            self.fit_zoom = min(available_width/self.image_width,available_height/self.image_height)
            self.zoom = min(MAX_ZOOM,max(MIN_ZOOM,self.fit_zoom))
            self.pan_x = self.pan_y = 0.0
            self.update_offsets()
        self.redraw()

    def update_offsets(self):
        self.image_offset_x = (self.canvas.winfo_width()-self.image_width*self.zoom)/2.0
        self.image_offset_y = (self.canvas.winfo_height()-self.image_height*self.zoom)/2.0

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

    def find_joint_at(self,x,y):
        radius = max(10,min(24,14*self.zoom/self.fit_zoom if self.fit_zoom > 0 else 14))
        hits = []
        for index,name in enumerate(JOINT_NAMES):
            if self.point_visibility[name].get():
                sx,sy = self.normalize_to_screen(self.joints[name])
                distance = math.hypot(sx-x,sy-y)
                if distance <= radius:
                    hits.append((distance,index,name))
        return min(hits)[2] if hits else None

    def get_descendants(self,joint_name):
        descendants,pending = [],[joint_name]
        while pending:
            current = pending.pop(0)
            children = [name for name,parent in PARENT_MAP.items() if parent == current]
            descendants += children
            pending += children
        return descendants

    def get_subtree_positions(self,joint_name):
        return self._positions([joint_name]+self.get_descendants(joint_name))

    def _rotate_points(self,positions,center_px,center_py,angle):
        cos_a,sin_a = math.cos(angle),math.sin(angle)
        for name,(x,y) in positions.items():
            dx,dy = x*self.image_width-center_px,y*self.image_height-center_py
            self.joints[name].x = (center_px+dx*cos_a-dy*sin_a)/self.image_width
            self.joints[name].y = (center_py+dx*sin_a+dy*cos_a)/self.image_height

    def rotate_subtree(self,joint_name,new_x,new_y):
        parent_name = PARENT_MAP.get(joint_name)
        if parent_name is None or parent_name not in self.joints:
            self.joints[joint_name].x,self.joints[joint_name].y = float(new_x),float(new_y)
            return
        if joint_name not in self.drag_reference_positions or self.image_width <= 0 or self.image_height <= 0:
            return
        parent = self.joints[parent_name]
        base_px,base_py = parent.x*self.image_width,parent.y*self.image_height
        start_x,start_y = self.drag_reference_positions[joint_name]
        start_px,start_py = start_x*self.image_width,start_y*self.image_height
        if math.hypot(start_px-base_px,start_py-base_py) <= 1e-9:
            return
        angle = math.atan2(new_y*self.image_height-base_py,new_x*self.image_width-base_px)-math.atan2(start_py-base_py,start_px-base_px)
        self._rotate_points({name:point for name,point in self.drag_reference_positions.items() if name != parent_name},base_px,base_py,angle)

    def on_mouse_down(self,event):
        self.last_mouse_x,self.last_mouse_y = event.x,event.y
        self.dragging = True
        joint_name = None if self.move_skeleton.get() else self.find_joint_at(event.x,event.y)
        if self.move_skeleton.get():
            self.selected_joint,self.drag_mode,cursor = "__ALL__","move_skeleton","fleur"
            self.drag_start_x,self.drag_start_y = event.x,event.y
            self.move_reference_positions = self._positions(JOINT_NAMES)
        elif joint_name is None:
            self.selected_joint,self.drag_mode,cursor = None,"pan_left","fleur"
            self.pan_last_x,self.pan_last_y = event.x,event.y
        else:
            self.selected_joint,self.drag_mode,cursor = joint_name,"free" if self.extendable.get() else "rigid","crosshair"
            self.drag_start_x,self.drag_start_y = event.x,event.y
            self.drag_reference_positions = self.get_subtree_positions(joint_name)
        self.canvas.configure(cursor=cursor)
        self.redraw()

    def on_mouse_drag(self,event):
        if not self.dragging:
            return
        if self.drag_mode == "pan_left":
            self.pan_x += event.x-self.last_mouse_x
            self.pan_y += event.y-self.last_mouse_y
        elif self.drag_mode == "move_skeleton":
            self.move_all_joints(event.x-self.drag_start_x,event.y-self.drag_start_y)
        elif self.drag_mode in ("free","rigid") and self.selected_joint in self.joints:
            x = self.screen_to_image_x(event.x)/self.image_width if self.image_width else 0.5
            y = self.screen_to_image_y(event.y)/self.image_height if self.image_height else 0.5
            if self.drag_mode == "free":
                self.joints[self.selected_joint].x,self.joints[self.selected_joint].y = x,y
            else:
                self.rotate_subtree(self.selected_joint,x,y)
        self.last_mouse_x,self.last_mouse_y = event.x,event.y
        self.redraw()

    def move_all_joints(self,screen_dx,screen_dy):
        if not self.move_reference_positions or self.zoom <= 0 or self.image_width <= 0 or self.image_height <= 0:
            return
        delta_x,delta_y = screen_dx/(self.zoom*self.image_width),screen_dy/(self.zoom*self.image_height)
        for name,(x,y) in self.move_reference_positions.items():
            self.joints[name].x,self.joints[name].y = x+delta_x,y+delta_y

    def _reset_drag(self):
        self.dragging,self.drag_mode = False,None
        self.drag_reference_positions,self.move_reference_positions = {},{}

    def on_mouse_up(self,event):
        self._reset_drag()
        self.canvas.configure(cursor="arrow")

    def on_middle_mouse_down(self,event):
        self.panning = True
        self.pan_last_x,self.pan_last_y = event.x,event.y
        self.canvas.configure(cursor="fleur")

    def on_middle_mouse_drag(self,event):
        if not self.panning:
            return
        self.pan_x += event.x-self.pan_last_x
        self.pan_y += event.y-self.pan_last_y
        self.pan_last_x,self.pan_last_y = event.x,event.y
        self.redraw()

    def on_middle_mouse_up(self,event):
        self.panning = False
        self.canvas.configure(cursor="arrow")

    def on_mousewheel(self,event):
        if self.image is None:
            return
        zoom_in = event.delta > 0 if getattr(event,"delta",0) else event.num == 4
        self.zoom_at_screen_point(event.x,event.y,1.12 if zoom_in else 1/1.12)

    def zoom_at_screen_point(self,sx,sy,factor):
        new_zoom = max(MIN_ZOOM,min(MAX_ZOOM,self.zoom*factor))
        if abs(new_zoom-self.zoom) < 1e-9:
            return
        ix,iy = self.screen_to_image_x(sx),self.screen_to_image_y(sy)
        self.zoom = new_zoom
        self.update_offsets()
        self.pan_x,self.pan_y = sx-self.image_offset_x-ix*self.zoom,sy-self.image_offset_y-iy*self.zoom
        self.redraw()

    def rotate_skeleton(self,degrees):
        visible = [name for name in JOINT_NAMES if self.point_visibility[name].get()]
        if not visible or self.image_width <= 0 or self.image_height <= 0:
            return
        center_px = sum(self.joints[name].x for name in visible)/len(visible)*self.image_width
        center_py = sum(self.joints[name].y for name in visible)/len(visible)*self.image_height
        self._rotate_points(self._positions(visible),center_px,center_py,math.radians(degrees))
        self.redraw()

    def reset_skeleton(self):
        for name,(x,y) in self.original_pose.items():
            self.joints[name].x,self.joints[name].y = x,y
        self.selected_joint = "__ALL__" if self.move_skeleton.get() else None
        self.redraw()

    def toggle_move_skeleton(self):
        self._reset_drag()
        moving = self.move_skeleton.get()
        self.selected_joint = "__ALL__" if moving else None
        self.canvas.configure(cursor="fleur" if moving else "arrow")
        self.redraw()

    def draw_grid(self):
        if not self.show_grid.get() or self.image is None:
            return
        width,height = self.image_width,self.image_height
        for i in range(11):
            ratio = i/10
            for start,end in (((ratio*width,0),(ratio*width,height)),((0,ratio*height),(width,ratio*height))):
                self.canvas.create_line(*self.image_to_screen(*start),*self.image_to_screen(*end),fill="#3a3a3a",width=1)

    def draw_skeleton_at_scale(self):
        for a,b in BONES:
            if self._visible(a,b):
                self.canvas.create_line(*self.normalize_to_screen(self.joints[a]),*self.normalize_to_screen(self.joints[b]),fill="#ffffff",width=4,capstyle=tk.ROUND)
        for name in JOINT_NAMES:
            if not self._visible(name):
                continue
            x,y = self.normalize_to_screen(self.joints[name])
            radius,outline = (6 if name in self.inferred_joints else 7),"#ffffff"
            if self.move_skeleton.get() or name == self.selected_joint:
                radius,outline = radius+2,"#ff0000"
            self.canvas.create_oval(x-radius,y-radius,x+radius,y+radius,fill=OPENPOSE_COLORS.get(name,"#ffffff"),outline=outline,width=2)
            if self.show_labels.get():
                self.canvas.create_text(x+10,y,text=name,anchor="w",fill="#ffffff",font=("Arial",9,"bold"))
        if self.show_angles.get():
            self.draw_angles()

    def draw_angles(self):
        for a,b,c in ANGLE_GROUPS:
            if not self._visible(a,b,c):
                continue
            p1,p2,p3 = self.joints[a],self.joints[b],self.joints[c]
            v1,v2 = np.array([p1.x-p2.x,p1.y-p2.y],dtype=float),np.array([p3.x-p2.x,p3.y-p2.y],dtype=float)
            n1,n2 = np.linalg.norm(v1),np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                continue
            angle = math.degrees(math.acos(max(-1.0,min(1.0,float(np.dot(v1,v2)/(n1*n2))))))
            sx,sy = self.normalize_to_screen(p2)
            self.canvas.create_text(sx+15,sy+15,text=f"{angle:.1f}°",fill="#00ffcc",font=("Arial",9,"bold"))

    def redraw(self):
        if not hasattr(self,"canvas"):
            return
        self.canvas.delete("all")
        if self.image is None:
            self.canvas.create_text(max(1,self.canvas.winfo_width()/2),max(1,self.canvas.winfo_height()/2),text="Upload an image to begin",fill="#888888",font=("Arial",18))
            return
        self.update_offsets()
        scaled_size = (max(1,int(round(self.image_width*self.zoom))),max(1,int(round(self.image_height*self.zoom))))
        self._tk_image = ImageTk.PhotoImage(self.dark_image.resize(scaled_size,Image.Resampling.LANCZOS))
        self.canvas.create_image(self.image_offset_x+self.pan_x,self.image_offset_y+self.pan_y,image=self._tk_image,anchor="nw")
        self.draw_grid()
        self.draw_skeleton_at_scale()

    def prepare_detection_image(self,image):
        scale = DETECTION_RESOLUTION/max(image.size)
        return image.resize((max(64,int(round(image.width*scale/64))*64),max(64,int(round(image.height*scale/64))*64)),Image.Resampling.LANCZOS)

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
        if not isinstance(pose,dict):
            return []
        if "body_pose" in pose:
            return pose["body_pose"]
        values = pose.get("pose_keypoints_2d")
        if "pose_keypoints_2d" in pose and isinstance(values,list) and len(values)%3 == 0:
            return [values[i:i+3] for i in range(0,len(values),3)]
        return pose

    def keypoint_is_valid(self,point):
        if point is None:
            return False
        try:
            if hasattr(point,"x") and hasattr(point,"y"):
                return math.isfinite(float(point.x)) and math.isfinite(float(point.y))
            if isinstance(point,(list,tuple)) and len(point) >= 2:
                x,y,score = float(point[0]),float(point[1]),float(point[2]) if len(point) >= 3 else 1.0
                return math.isfinite(x) and math.isfinite(y) and score > 0.05
        except Exception:
            return False
        return False

    def get_keypoint_xy(self,point):
        return (float(point.x),float(point.y)) if hasattr(point,"x") and hasattr(point,"y") else (float(point[0]),float(point[1]))

    def keypoints_to_points(self,pose,source_width,source_height):
        points = {}
        keypoints = self.get_pose_body_keypoints(pose)
        if keypoints is None:
            return points
        if isinstance(keypoints,np.ndarray):
            keypoints = keypoints.tolist()
        for index,name in enumerate(JOINT_NAMES[:len(keypoints)]):
            point = keypoints[index]
            if not self.keypoint_is_valid(point):
                continue
            x,y = self.get_keypoint_xy(point)
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                x,y = x/float(source_width),y/float(source_height)
            points[name] = (self.clamp01(x),self.clamp01(y))
        return points

    def infer_knee_on_leg_line(self,points,side):
        hip,knee,ankle = f"{side}Hip",f"{side}Knee",f"{side}Ankle"
        if knee in points or hip not in points or ankle not in points:
            return None
        (hx,hy),(ax,ay) = points[hip],points[ankle]
        return self.clamp01(hx+(ax-hx)*KNEE_LINE_RATIO),self.clamp01(hy+(ay-hy)*KNEE_LINE_RATIO)

    def smart_complete_points(self,points):
        result,inferred = dict(points),set()
        for right,left in MIRROR_PAIRS:
            for known,missing in ((right,left),(left,right)):
                if known in result and missing not in result:
                    x,y = result[known]
                    result[missing] = (self.clamp01(1.0-x),y)
                    inferred.add(missing)
                    break
        for side in ("R","L"):
            knee = self.infer_knee_on_leg_line(result,side)
            if knee is not None:
                result[f"{side}Knee"] = knee
                inferred.add(f"{side}Knee")
        return result,inferred

    def select_best_pose(self,poses,source_width,source_height):
        best_pose,best_score = None,(-1,-1,-1)
        for pose in poses:
            valid_count = len(self.keypoints_to_points(pose,source_width,source_height))
            total_score,parts = 0.0,len(self.get_pose_body_keypoints(pose))
            if hasattr(pose,"body"):
                try:
                    total_score = float(getattr(pose.body,"total_score",0.0))
                except Exception:
                    total_score = 0.0
                try:
                    parts = int(getattr(pose.body,"total_parts",parts))
                except Exception:
                    pass
            if (valid_count,total_score,parts) > best_score:
                best_score,best_pose = (valid_count,total_score,parts),pose
        return best_pose

    def _set_pose(self,points):
        for name in JOINT_NAMES:
            found = name in points
            self.joints[name].x,self.joints[name].y = points[name] if found else DEFAULT_POSE[name]
            self.point_visibility[name].set(found)
        self.original_pose = self._positions(JOINT_NAMES)

    def apply_smart_detected_pose(self,smart_points,inferred):
        self.inferred_joints = set(inferred)
        self._set_pose(smart_points)
        self.selected_joint = None

    def detect_pose(self,image):
        self.load_openpose_detector()
        detection_image = self.prepare_detection_image(image)
        poses = self.detector.detect_poses(np.array(detection_image,dtype=np.uint8),include_hand=False,include_face=False)
        if not poses:
            raise PoseNotFound("OpenPose did not detect any body pose.")
        best_pose = self.select_best_pose(poses,*detection_image.size)
        if best_pose is None:
            raise PoseNotFound("Unable to select a detected body pose.")
        points = self.keypoints_to_points(best_pose,*detection_image.size)
        if len(points) < MIN_VALID_KEYPOINTS:
            raise PoseNotFound(f"OpenPose detected only {len(points)} valid body points.")
        self.apply_smart_detected_pose(*self.smart_complete_points(points))

    def initialize_pose_from_original(self):
        if self.original_image is None:
            return
        try:
            self.detect_pose(self.original_image)
            self.redraw()
        except PoseNotFound:
            pass
        except Exception:
            traceback.print_exc()

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files","*.png *.jpg *.jpeg *.webp *.bmp"),("All Files","*.*")])
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
            self.detect_pose(image)
            self.set_editor_image(image)
            self.redraw()
        except Exception as error:
            traceback.print_exc()
            messagebox.showerror("OpenPose Detection Error",f"{error}")

    def load_skeleton_json(self):
        path = filedialog.askopenfilename(filetypes=JSON_TYPES)
        if not path:
            return
        try:
            with open(path,"r",encoding="utf-8") as file:
                self.apply_pose_json(json.load(file))
            self.redraw()
            messagebox.showinfo("Skeleton Loaded",f"Loaded skeleton JSON:\n{path}")
        except Exception as error:
            traceback.print_exc()
            messagebox.showerror("Skeleton JSON Error",f"Could not load skeleton JSON.\n\n{error}")

    def save_skeleton_json(self):
        try:
            pose_data = self._visible_pose_json()
            path = filedialog.asksaveasfilename(defaultextension=".json",filetypes=JSON_TYPES)
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
        source_width,source_height = float(pose_data.get("image_width",0)),float(pose_data.get("image_height",0))
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Skeleton JSON image dimensions must be positive.")
        for name,point in joints.items():
            if name not in JOINT_NAMES:
                continue
            if not isinstance(point,(list,tuple)) or len(point) < 2:
                raise ValueError(f'Joint "{name}" must contain [x,y].')
            if not math.isfinite(float(point[0])) or not math.isfinite(float(point[1])):
                raise ValueError(f'Joint "{name}" contains an invalid coordinate.')
        return source_width,source_height

    def apply_pose_json(self,pose_data):
        source_width,source_height = self.validate_pose_json(pose_data)
        if self.image is None and self.original_image is not None:
            self.set_editor_image(self.original_image.copy())
        if self.image is None:
            raise RuntimeError("Upload an image before loading a skeleton JSON.")
        joints = pose_data.get("joints",{})
        self.inferred_joints = set()
        self._set_pose({name:(float(point[0])/source_width,float(point[1])/source_height) for name in JOINT_NAMES if (point := joints.get(name)) is not None})
        self.selected_joint = "__ALL__" if self.move_skeleton.get() else None

    def build_pose_json(self):
        if self.image is None:
            raise RuntimeError("No editor image is loaded.")
        joints = {name:[round(float(self.joints[name].x*self.image_width),4),round(float(self.joints[name].y*self.image_height),4)] for name in JOINT_NAMES if self.point_visibility[name].get()}
        return {"format":"OpenPose-Stickman-Rigid","version":29,"source":"pose_editor.py","image_width":int(self.image_width),"image_height":int(self.image_height),"joints":joints,"bones":[{"a":a,"b":b} for a,b in BONES if a in joints and b in joints]}

    def _visible_pose_json(self):
        pose_data = self.build_pose_json()
        if not pose_data.get("joints"):
            raise RuntimeError("No visible skeleton joints are available.")
        return pose_data

    def apply_skeleton(self):
        try:
            pose_data = self._visible_pose_json()
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
