# SimpleDiffusion

A native UI for Stable Diffusion using the `diffusers` library.  Developed and tested on macOS.  Other platforms could be supported with a small effort.

![Screenshot](docs/screenshot.webp)

### Installation

1. Open Terminal.
2. Navigate to the directory containing this project.
3. Execute the install script (use this also to update when requirements/dependency versions change)
    ```sh
    ./install.sh
    ```
4. Run the application
    ```sh
    ./run.sh
    ```

### Features

- Text to Image
- Image to Image
- ControlNet 1.1 (Base, Img2Img, Multi-model)
- LoRA
- Textual Inversion
- High-resolution fix
- ESRGAN Upscaling
- GFPGAN Face Restoration
- Image generation with real-time preview of latent space
- Thumbnail viewer with collections

### Roadmap

Near-term:
- Inpainting
- Async thumbnail loading
- Upscale/Restore Faces for existing images
- ControlNet preprocessor arguments
- Thumbnail paths by image hash 
- Collapsing panels in UI

Evaluate:
- Self-Attention Guidance
- MultiDiffusion
- Long Prompt Weighting
- ControlNet Guess Mode

Requires new versions of dependencies:
- torch
  - Retest float16 support
