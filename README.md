# SimpleDiffusion

A simple Python GUI for the `diffusers` library.  Inspired by InvokeAI, but uses less memory making it faster to run on a machine with low RAM (e.g. 16GB).  Developed and tested on macOS.  Other platforms could be supported with a small effort.

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
    pip install accelerate
    pip install diffusers
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

- Samplers (hard-coded to Euler Ancestral)
- Pipeline cancellation
- Keyboard shortcuts
- Models (hard-coded in source)
- Upscaling ESRGAN
- Image collections
- Thumbnail Cache
- In Progress Display
- Additional pipelines (e.g. depth2img, ControlNet)
- Local models
- Respond to local filesystem changes
