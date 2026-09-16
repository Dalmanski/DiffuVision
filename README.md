# DiffuVision

> **Work in Progress (WIP)** — This application is not yet available as a standalone `.exe`.

## Setup

To run **DiffuVision AI**, execute the Python application using **Visual Studio Code** or your preferred terminal.

> **Hardware Recommendation:** An NVIDIA GPU with CUDA support is strongly recommended. CPU-only execution is currently untested and may result in significantly slower performance.
I used on RTX 2050, 4GB VRAM and 16GB RAM.

### 1. Download the Project

Clone or download this repository, then open the project folder in **Visual Studio Code**.

### 2. Download the Required Models

Download one or both of the following inpainting models:

* **DreamShaper 8 Inpainting**

  * **Link:** [DreamShaper_8_INPAINTING.inpainting.safetensors](https://huggingface.co/Lykon/DreamShaper/blob/main/DreamShaper_8_INPAINTING.inpainting.safetensors)
  * **Best for:** General-purpose use, including real-life and anime images

**OR**

* **LazyMix v4.0 Inpainting**

  * **Link:** [lazymixRealAmateur_v40Inpainting.safetensors](https://huggingface.co/TheImposterImposters/LazyMix-v4.0-inpainting/blob/main/lazymixRealAmateur_v40Inpainting.safetensors)
  * **Best for:** Photorealistic and real-life images

### 3. Place the Models

After downloading the models, place them in:

```text
DiffuVision AI/model/
```

The folder structure should look similar to this:

```text
DiffuVision AI/
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

### 4. Download SAM2

Download **SAM2**, including `sam2.1_hiera_tiny.pt`.

Rename the downloaded `sam2` folder to `sam2_repo` and place it within the project directory as shown below:

```text
DiffuVision AI/
├── modules/
│   ├── sam2_repo/
│   │   ├── sam2.1_hiera_tiny.pt
│   │   └── ...
│   └── ...
├── app.py
└── ...
```

### 5. Run the Application

Open the project in **Visual Studio Code** and run:

```bash
py app.py
```

The application will automatically download and install the required Python dependencies when necessary.

## Notes & Updates

This project is actively under development and does not currently have a standalone executable (`.exe`).

* **Date Created:** September 1, 2026
* **Status:** Work in Progress (WIP)