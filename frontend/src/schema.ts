import { v4 as uuid4 } from "uuid";
import {
  ControlNetConditionRequest,
  FaceRestorationRequest,
  HighResRequest,
  ImageRequest,
  Img2ImgRequest,
  LoraModelRequest,
  RefinerRequest,
  UpscaleRequest,
} from "./requests.ts";
import { loadArray, loadNew, loadNumberArray, loadOptional, loadProps } from "./util/loadUtil.ts";
import { Vec2 } from "./vec2.ts";

const imageRequest = new ImageRequest();
const img2imgRequest = new Img2ImgRequest();
const loraModelRequest = new LoraModelRequest();
const conditionRequest = new ControlNetConditionRequest();
const refinerRequest = new RefinerRequest();
const upscaleRequest = new UpscaleRequest();
const faceRequest = new FaceRestorationRequest();
const highResRequest = new HighResRequest();

export class PromptParamsState {
  isOpen: boolean = true;
  prompt: string = imageRequest.prompt;
  negativePrompt: string = imageRequest.negative_prompt;

  load(src: Partial<PromptParamsState>) {
    loadProps(this, src);
    return this;
  }
}

export class GeneralParamsState {
  isOpen: boolean = true;
  model: string = imageRequest.model;
  scheduler: string = imageRequest.scheduler;
  imageCount: number = imageRequest.image_count;
  steps: number = imageRequest.steps;
  cfgScale: number = imageRequest.cfg_scale;
  width: number = imageRequest.width;
  height: number = imageRequest.height;

  load(src: Partial<GeneralParamsState>) {
    loadProps(this, src);
    return this;
  }
}

export class SeedParamsState {
  isOpen: boolean = false;
  isEnabled: boolean = false;
  seed: number = imageRequest.seed;

  load(src: Partial<SeedParamsState>) {
    loadProps(this, src);
    return this;
  }
}

export class Img2ImgParamsState {
  isOpen: boolean = false;
  isEnabled: boolean = false;
  source: string = img2imgRequest.source;
  noise: number = img2imgRequest.noise;

  load(src: Partial<Img2ImgParamsState>) {
    loadProps(this, src);
    return this;
  }
}

export class LoraModelParamsState {
  id: string = uuid4();
  model: string = loraModelRequest.model;
  weight: number = loraModelRequest.weight;

  load(src: Partial<LoraModelParamsState>) {
    loadProps(this, src);
    return this;
  }
}

export class LoraParamsState {
  isOpen: boolean = false;
  isEnabled: boolean = false;
  entries: LoraModelParamsState[] = [];

  load(src: Partial<LoraParamsState>) {
    loadProps(this, src);
    loadArray(this.entries, src.entries, LoraModelParamsState);
    return this;
  }
}

export class ControlNetConditionParamsState {
  id: string = uuid4();
  isOpen: boolean = false;
  isEnabled: boolean = false;
  guidanceStart: number = conditionRequest.guidance_start;
  guidanceEnd: number = conditionRequest.guidance_end;
  model: string = conditionRequest.model;
  source: string = conditionRequest.source;
  processor: string = conditionRequest.processor;
  params: Map<string, number> = new Map<string, number>();
  scale: number = conditionRequest.scale;

  load(src: Partial<ControlNetConditionParamsState>) {
    loadProps(this, src);
    // TODO - params
    return this;
  }
}

export class ControlNetParamsState {
  isOpen: boolean = false;
  isEnabled: boolean = false;
  activeTab: number = 0;
  conditions: ControlNetConditionParamsState[] = [];

  load(src: Partial<ControlNetParamsState>) {
    loadProps(this, src);
    loadArray(this.conditions, src.conditions, ControlNetConditionParamsState);
    return this;
  }
}

export class RefinerParamsState {
  isOpen: boolean = false;
  isEnabled: boolean = false;
  ensembleMode: boolean = false;
  cfgScale: number = refinerRequest.cfg_scale;
  highNoiseEnd: number = 0.8;
  steps: number = 20;
  noise: number = 0.3;

  load(src: Partial<RefinerParamsState>) {
    loadProps(this, src);
    return this;
  }
}

export class UpscaleParamsState {
  isOpen: boolean = false;
  isEnabled: boolean = false;
  factor: number = upscaleRequest.factor;
  denoising: number = upscaleRequest.denoising;
  blend: number = upscaleRequest.blend;

  load(src: Partial<UpscaleParamsState>) {
    loadProps(this, src);
    return this;
  }
}

export class FaceParamsState {
  isOpen: boolean = false;
  isEnabled: boolean = false;
  blend: number = faceRequest.blend;

  load(src: Partial<FaceParamsState>) {
    loadProps(this, src);
    return this;
  }
}

export class HighResParamsState {
  isOpen: boolean = false;
  isEnabled: boolean = false;
  factor: number = highResRequest.factor;
  steps: number = highResRequest.steps;
  cfgScale: number = highResRequest.cfg_scale;
  noise: number = highResRequest.noise;

  load(src: Partial<HighResParamsState>) {
    loadProps(this, src);
    return this;
  }
}

export class InpaintParamsState {
  isOpen: boolean = false;
  isEnabled: boolean = false;
  source: string = "";
  useAlphaChannel: boolean = false;
  invertMask: boolean = false;

  load(src: Partial<InpaintParamsState>) {
    loadProps(this, src);
    return this;
  }
}

export class GenerationParamsState {
  prompt = new PromptParamsState();
  general = new GeneralParamsState();
  seed = new SeedParamsState();
  img2img = new Img2ImgParamsState();
  lora = new LoraParamsState();
  controlNet = new ControlNetParamsState();
  refiner = new RefinerParamsState();
  upscale = new UpscaleParamsState();
  face = new FaceParamsState();
  highRes = new HighResParamsState();
  inpaint = new InpaintParamsState();

  load(src: Partial<GenerationParamsState>) {
    loadProps(this, src);
    this.prompt = loadNew(src.prompt, PromptParamsState);
    this.general = loadNew(src.general, GeneralParamsState);
    this.seed = loadNew(src.seed, SeedParamsState);
    this.img2img = loadNew(src.img2img, Img2ImgParamsState);
    this.lora = loadNew(src.lora, LoraParamsState);
    this.controlNet = loadNew(src.controlNet, ControlNetParamsState);
    this.refiner = loadNew(src.refiner, RefinerParamsState);
    this.upscale = loadNew(src.upscale, UpscaleParamsState);
    this.face = loadNew(src.face, FaceParamsState);
    this.highRes = loadNew(src.highRes, HighResParamsState);
    this.inpaint = loadNew(src.inpaint, InpaintParamsState);
    return this;
  }
}

export class CanvasImage {
  path: string = "";

  load(src: Partial<CanvasImage>) {
    loadProps(this, src);
    return this;
  }
}

export class CanvasElementState {
  id: string = uuid4();
  x: number = 0;
  y: number = 0;
  width: number = 512;
  height: number = 512;
  generation: GenerationParamsState | null = null;
  images: CanvasImage[] = [];
  imageIndex: number = 0;

  load(src: Partial<CanvasElementState>) {
    loadProps(this, src);
    this.generation = loadOptional(src.generation, GenerationParamsState);
    loadArray(this.images, src.images, CanvasImage);
    return this;
  }
}

export class CanvasStrokeState {
  tool: string = "select";
  segments: number[] = [];

  load(src: Partial<CanvasStrokeState>) {
    loadProps(this, src);
    loadNumberArray(this.segments, src.segments);
    return this;
  }
}

// TODO - persist on server, local for now
export class CanvasModeState {
  scale: number = 1;
  translate: Vec2 = Vec2.create(0, 0);
  cursorPos: Vec2 | null = null;
  tool: string = "select";
  elements: CanvasElementState[] = [];
  strokes: CanvasStrokeState[] = [];
  selectedId: string | null = null;
  hoveredId: string | null = null;

  load(src: Partial<CanvasModeState>) {
    loadProps(this, src);
    this.translate = loadNew(src.translate, Vec2);
    this.cursorPos = loadOptional(src.cursorPos, Vec2);
    loadArray(this.elements, src.elements, CanvasElementState);
    loadArray(this.strokes, src.strokes, CanvasStrokeState);
    return this;
  }
}

// Does not persist
export class SessionState {
  sessionId: string = "";
  selectedIndex: number = 0;
  generatorId: string | null = null;
  progressAmount: number = 0;
  previewUrl: string | null = null;
  historyStack: string[] = [];
  historyStackIndex: number = -1;
  dialog: string | null = null;
  deleteImagePath: string = "";

  load(src: Partial<SessionState>) {
    loadProps(this, src);
    return this;
  }
}

// Persists on server
export class SettingsState {
  safetyChecker: boolean = true;
  collection: string = "outputs";
  generation = new GenerationParamsState();
  showMetadata: boolean = false;
  showPreview: boolean = false;

  load(src: Partial<SettingsState>) {
    loadProps(this, src);
    this.generation = loadNew(src.generation, GenerationParamsState);
    return this;
  }
}

// Persists in local storage
export class SystemState {
  user: string = "default";
  mode: string = "image";

  load(src: Partial<SystemState>) {
    loadProps(this, src);
    return this;
  }
}
