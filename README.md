<div align="center">

![Logo](docs/logo.webp)

# SeedAlchemy: AI Image Generation

![Screenshot](docs/screenshot.webp)

</div>

## Features

- Text to Image
- Image to Image
- Prompt Weighting using [Compel](https://github.com/damian0815/compel/blob/main/Reference.md)
- ControlNet (1.0, 1.1, Txt2Img, Img2Img, Multi-model)
- LoRA
- Textual Inversion
- High-resolution fix
- ESRGAN Upscaling
- GFPGAN Face Restoration
- Prompt Generator
- Clip Interrogator
- Real-time preview of latent space
- Thumbnail viewer with collections
- Canvas mode (IN DEVELOPMENT)

## Installation

### Windows

1. Install [Python](https://www.python.org/downloads/windows/), checking "Add Python to PATH".
2. Install [git](https://git-scm.com/download/win).
3. Download this repository, for example by running `git clone https://github.com/pdoane/seed-alchemy.git`.
4. Execute `install.bat` from Windows Explorer as normal, non-administrator, user.
   Use this also to update when requirements/dependency versions change
5. Execute `run.bat` from Windows Explorer as normal, non-administrator, user.

### macOS

1. Install Python and git, one option is via the MacOS Developer Tools
2. Download this repository, for example by running `git clone https://github.com/pdoane/seed-alchemy.git`.
3. Navigate a terminal to the directory containing this project.
4. Execute `install.sh` (use this also to update when requirements/dependency versions change)
5. Execute `run.sh`

## Roadmap

Near-term:
- Canvas Mode
  - Painting
    - Sketch
    - Brush size
    - Color picker
    - Eye dropper
    - Selection
      - Marquee
      - Lasso
      - Smart mode: Foreground/background/prompt
      - Blur
      - All/None/Invert
    - Fill/Clear
    - Undo
  - Generation
    - Image
    - Outpainting
    - Inpainting
    - Sketch
    - Blending
  - Elements
    - Stack UI
    - Reordering
    - Upscale/downscale
    - Collapse
- Image Browser
- Project files (replaces --root)
- Replace collections with directory hierarchy
- Image Gallery
- PNG/JPEG metadata viewer
- ControlNet
  - Segmentation models
  - Reference-Only control
  - Control Mode (previously Guess Mode)
- Long Prompt support
- A1111 style weighting 

Evaluate:
- ControlNet T2I-Adapter
- ControlNet High-resolution control image sampling
