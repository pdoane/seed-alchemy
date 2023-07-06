<div align="center">

![Logo](docs/logo.webp)

# SeedAlchemy: AI Image Generation

![Screenshot](docs/screenshot.webp)

</div>

## Features

- Text to Image
- Image to Image
- Inpainting
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
- Image gallery/slideshow mode
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

QT performance on Windows for the Canvas feature is not as I good as I would like. So I have put development on hold for a moment to test other technology stacks. I have a basic prototype complete now using React and expect to switch to that soon.
