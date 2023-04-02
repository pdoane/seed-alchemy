# SimpleDiffusion

A simple Python GUI for the `diffusers` library.  Inspired by InvokeAI, but uses less memory making it faster to run on a machine with low RAM (e.g. 16GB).  Developed and tested on macOS.  Other platforms could be supported with a small effort.

![Screenshot](data/screenshot.webp)

### Installation

1. Open Terminal.
2. Navigate to the directory containing this project.
3. Create a virtual environment named `.venv` inside this directory and activate it:
    ```sh
    python3 -m venv .venv --prompt SimpleDiffusion
    ```
4. Activate the virtual environment (do it every time you run SimpleDiffusion)
    ```sh
    source .venv/bin/activate
    ```
5. Install dependencies.
    ```sh
    curl -L -o data/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
    pip install diffusers
    pip install accelerate
    pip install safetensors
    pip install PySide6
    pip install numpy
    pip install torch
    pip install compel
    pip install gfpgan
    pip install pyobjc
    ```
6. Run the GUI
    ```sh
    python qtgui.py
    ```

### Roadmap

- Program Settings UI
- Source image: Unset on Delete, improve display when empty, find in thumbnails
- Additional pipelines (e.g. depth2img, ControlNet)
- Local models
- Long Prompt Weighting
- Latent Upscaling (revisit later, issues to resolve in diffusers library)
