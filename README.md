<div align="center">

![Logo](docs/logo.webp)

# SeedAlchemy: AI Image Generation

![Screenshot](docs/screenshot.webp)

</div>

## Features

- Stable Diffusion 1.x, 2.x, SDXL
- Text to Image
- Image to Image
- Inpainting
- Prompt Weighting using [Compel](https://github.com/damian0815/compel/blob/main/Reference.md)
- ControlNet
- LoRA
- Textual Inversion/Embeddings
- High-resolution fix
- ESRGAN Upscaling
- GFPGAN Face Restoration
- Prompt Generator
- Real-time preview of latent space
- Thumbnail viewer with collections

## Installation

1. Install Python and Node.js as prerequisites.
2. Execute `install.sh` or `install.bat`. Also use these scripts when dependencies change.

## Running

Backend:

- In the top-level directory, execute `backend/run.sh` or `backend\run.bat`.

Frontend:

- In the `frontend` directory:
- `npm run dev` for development in a web browser
- `npm run build` to compile .js to be served from the backend
- Electron support is available with `npm run electron-dev` and `npm run electron-build`

## Roadmap

In progress switching technology stacks to use React for the frontend. Features from the earlier QT version to still implement:

- Clip Interrogator
- Image gallery/slideshow mode
- Canvas mode (IN DEVELOPMENT)
