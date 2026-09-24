# DiffuVision

> **Work in Progress (WIP)** — This application is not yet available as a standalone `.exe`.

## Disclaimer

**Use DiffuVision responsibly.** The developer is not responsible for misuse of this software or for inappropriate, harmful, or unlawful content created with it. Users are responsible for complying with applicable laws, obtaining necessary permissions, and following the licenses of any third-party models or software.

## Setup

To run **DiffuVision**, execute the Python application using **Visual Studio Code** or your preferred terminal.

> **Hardware Recommendation:** An NVIDIA GPU with CUDA support is strongly recommended. CPU-only execution is currently untested and may result in significantly slower performance.

I tested DiffuVision on an **RTX 2050 with 4 GB VRAM and 16 GB RAM**.

### 1. Download the Project

Download and install Python 3.12:

https://www.python.org/downloads/windows/

During installation, enable:

```text
Add Python to PATH
```

Then open a new VS Code terminal:

```powershell
python --version
```

If needed:

```text
Ctrl+Shift+P → Python: Select Interpreter → Python 3.12
```

Clone or download this repository, then open the project folder in **Visual Studio Code**.

### Install the Python Dependencies

Open the **VS Code terminal** inside your DiffuVision project folder and run:

```powershell
py -m pip install -r requirements.txt
```

## 2. Download the Required Models

Download the following inpainting models.

### DreamShaper 8 Inpainting

* **File:** `DreamShaper_8_INPAINTING.inpainting.safetensors`
* **Link:** https://huggingface.co/Lykon/DreamShaper/blob/main/DreamShaper_8_INPAINTING.inpainting.safetensors
* **Best for:** General-purpose use, including real-life and anime images.

## 3. Place the Models

After downloading the models, place them in:

```text
DiffuVision/model/
```

The folder structure should look similar to this:

```text
DiffuVision/
├── model/
│   └── DreamShaper_8_INPAINTING.inpainting.safetensors
├── app.py
└── ...
```

### Alternative: Use File Paths with a `.env` File

Instead of placing the models inside the `model` folder, you can specify their full file paths using a `.env` file.

Create a file named:

```text
.env
```

Then add:

```env
SD_INPAINT_MODEL='["YOUR FULL PATH TO DreamShaper_8_INPAINTING.inpainting.safetensors"]'
```

Replace the placeholder path with the actual location of the model file on your computer.

## Add a LoRA Model (Optional)

The project supports SD 1.5 LoRA files through the `SD_15_LoRA_MODEL` setting in `.env`.

### Recommended LoRA

This is a good starting point for extra detail/style control:

* **Model:** Detail Tweaker
* **Link:** https://civitai.com/models/58390/detail-tweaker-lora-lora
* **Recommended usage:** Same as a normal inpainting LoRA; keep it lightweight and test one model at a time.

Download the LoRA file (`.safetensors` or `.bin`) and place it in a folder such as:

```text
DiffuVision/model/LoRA/
```

Example structure:

```text
DiffuVision/
├── model/
│   ├── DreamShaper_8_INPAINTING.inpainting.safetensors
│   └── LoRA/
│       └── detail_tweaker.safetensors
├── app.py
└── ...
```

Then add the LoRA path to your project `.env` file:

```env
SD_15_LoRA_MODEL='["model/LoRA/detail_tweaker.safetensors"]'
```

You can also use a full absolute Windows path:

```env
SD_15_LoRA_MODEL='["C:/Users/YourName/Downloads/detail_tweaker.safetensors"]'
```

> **Important:** The app reads LoRA paths from `SD_15_LoRA_MODEL` in `.env`. If you want the LoRA to load, make sure the file exists and the path is correct.

## 4. Download and Install SAM 2.1

Open the **VS Code terminal** inside your DiffuVision project folder and run these commands **one at a time**:

```powershell
git clone https://github.com/facebookresearch/sam2.git modules\sam2_repo
cd modules\sam2_repo
pip install -e .
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" -OutFile "checkpoints\sam2.1_hiera_small.pt"
```

These commands will:

1. Download the official SAM 2 repository.
2. Install SAM 2.
3. Download the **SAM 2.1 Hiera Small** model.

Your project should look similar to this:

```text
DiffuVision/
├── modules/
│   ├── sam2_repo/
│   │   ├── checkpoints/
│   │   │   └── sam2.1_hiera_small.pt
│   │   ├── configs/
│   │   ├── sam2/
│   │   └── ...
│   └── ...
├── app.py
└── ...
```

> **Note:** The SAM 2.1 repository does not include the model weights when cloned. The last command downloads the `sam2.1_hiera_small.pt` checkpoint separately.

## 5. Run the Application

Open the project in **Visual Studio Code** and run:

```bash
py app.py
```

The application may automatically download and install required components or model files when necessary, depending on the current project configuration.

## Notes & Updates

This project is actively under development and does not currently have a standalone executable (`.exe`).

* **Date Created:** September 1, 2026
* **Status:** Work in Progress (WIP)

## My available for now:
### Model
* DreamShaper8
* lazymixRealAmateur
* realisticVisionV60B1
### LoRA
* [detail-tweaker-lora.safetensors](https://civitai.com/models/58390)
* [hourglassv01.safetensors](https://civitai.com/models/129130)
* [perfectb.safetensors](https://civitai.com/models/662593)
