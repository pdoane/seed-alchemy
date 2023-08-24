import { loadOptional, loadProps, loadArray } from "./util/loadUtil";

export class Img2ImgRequest {
  source: string = "";
  noise: number = 0.5;

  load(src: Partial<Img2ImgRequest>) {
    loadProps(this, src);
    return this;
  }
}

export class LoraModelRequest {
  model: string = "";
  weight: number = 0.5;

  load(src: Partial<LoraModelRequest>) {
    loadProps(this, src);
    return this;
  }
}

export class LoraRequest {
  entries: LoraModelRequest[] = [];

  load(src: Partial<LoraRequest>) {
    loadProps(this, src);
    loadArray(this.entries, src.entries, LoraModelRequest);
    return this;
  }
}

export class ControlNetConditionRequest {
  model: string = "";
  source: string = "";
  processor: string = "none";
  params: Map<string, number> = new Map<string, number>();
  guidance_start: number = 0.0;
  guidance_end: number = 1.0;
  scale: number = 1.0;

  load(src: Partial<ControlNetConditionRequest>) {
    loadProps(this, src);
    // TODO - params
    return this;
  }
}

export class ControlNetRequest {
  conditions: ControlNetConditionRequest[] = [];

  load(src: Partial<ControlNetRequest>) {
    loadProps(this, src);
    loadArray(this.conditions, src.conditions, ControlNetConditionRequest);
    return this;
  }
}

export class RefinerRequest {
  cfg_scale: number = 4.0;
  high_noise_end?: number = undefined;
  steps?: number = undefined;
  noise?: number = undefined;

  load(src: Partial<RefinerRequest>) {
    loadProps(this, src);
    return this;
  }
}

export class UpscaleRequest {
  factor: number = 2;
  denoising: number = 0.75;
  blend: number = 0.75;

  load(src: Partial<UpscaleRequest>) {
    loadProps(this, src);
    return this;
  }
}

export class FaceRestorationRequest {
  blend: number = 0.75;

  load(src: Partial<FaceRestorationRequest>) {
    loadProps(this, src);
    return this;
  }
}

export class HighResRequest {
  factor: number = 1.5;
  steps: number = 20;
  cfg_scale: number = 4.0;
  noise: number = 0.5;

  load(src: Partial<HighResRequest>) {
    loadProps(this, src);
    return this;
  }
}

export class InpaintRequest {
  source: string = "";
  use_alpha_channel: boolean = false;
  invert_mask: boolean = false;

  load(src: Partial<InpaintRequest>) {
    loadProps(this, src);
    return this;
  }
}

export class ImageRequest {
  session_id: string | null = null;
  generator_id: string | null = null;
  user: string = "";
  collection: string = "outputs";
  image_count: number = 1;
  model: string = "stable-diffusion-v1-5";
  scheduler: string = "euler_a";
  safety_checker: boolean = true;
  prompt: string = "";
  negative_prompt: string = "";
  steps: number = 20;
  cfg_scale: number = 4.0;
  width: number = 512;
  height: number = 512;
  seed: number = 1;
  img2img: Img2ImgRequest | null = null;
  lora: LoraRequest | null = null;
  control_net: ControlNetRequest | null = null;
  refiner: RefinerRequest | null = null;
  upscale: UpscaleRequest | null = null;
  face: FaceRestorationRequest | null = null;
  high_res: HighResRequest | null = null;
  inpaint: InpaintRequest | null = null;

  load(src: Partial<ImageRequest>) {
    loadProps(this, src);
    this.img2img = loadOptional(src.img2img, Img2ImgRequest);
    this.lora = loadOptional(src.lora, LoraRequest);
    this.control_net = loadOptional(src.control_net, ControlNetRequest);
    this.refiner = loadOptional(src.refiner, RefinerRequest);
    this.upscale = loadOptional(src.upscale, UpscaleRequest);
    this.face = loadOptional(src.face, FaceRestorationRequest);
    this.high_res = loadOptional(src.high_res, HighResRequest);
    this.inpaint = loadOptional(src.inpaint, InpaintRequest);
    return this;
  }
}
