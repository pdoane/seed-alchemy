// Default parameter values

import type { ImageParams } from "./types/image.js";

export const defaultParams: ImageParams = {
  prompt: "cutest cat in the world",
  negativePrompt: "",
  model: "",
  sampler: "euler",
  scheduler: "normal",
  width: 512,
  height: 512,
  steps: 20,
  cfgScale: 7,
  seed: 1,
  loras: [],
  sourceImage: "",
  sourceImageStrength: 0.75,
  referenceImages: [],
  referenceWeight: 1,
  referenceWeightType: "linear",
  referenceCombineMode: "concat",
  controlNets: [],
  faceDetailer: false,
  upscaleEnabled: false,
  upscaleFactor: 2,
};
