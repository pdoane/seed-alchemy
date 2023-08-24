import {
  ControlNetConditionRequest,
  ControlNetRequest,
  FaceRestorationRequest,
  HighResRequest,
  ImageRequest,
  Img2ImgRequest,
  InpaintRequest,
  LoraModelRequest,
  LoraRequest,
  RefinerRequest,
  UpscaleRequest,
} from "./requests.ts";
import { ControlNetConditionParamsState, GenerationParamsState, LoraModelParamsState } from "./schema";

function stateToLoraEntry(entry: LoraModelParamsState): LoraModelRequest {
  return new LoraModelRequest().load({
    model: entry.model,
    weight: entry.weight,
  });
}

function stateToControlNetConditionRequest(condition: ControlNetConditionParamsState): ControlNetConditionRequest {
  return new ControlNetConditionRequest().load({
    model: condition.model,
    source: condition.source,
    processor: condition.processor,
    params: condition.params,
    guidance_start: condition.guidanceStart,
    guidance_end: condition.guidanceEnd,
    scale: condition.scale,
  });
}

export function stateToImageRequest(
  generation: GenerationParamsState,
  sessionId: string,
  generatorId: string | null,
  user: string,
  collection: string,
  safetyChecker: boolean
): ImageRequest {
  const req = new ImageRequest().load({
    session_id: sessionId,
    generator_id: generatorId,
    user: user,
    collection: collection,
    image_count: generation.general.imageCount,
    model: generation.general.model,
    scheduler: generation.general.scheduler,
    safety_checker: safetyChecker,
    prompt: generation.prompt.prompt,
    negative_prompt: generation.prompt.negativePrompt,
    steps: generation.general.steps,
    cfg_scale: generation.general.cfgScale,
    width: generation.general.width,
    height: generation.general.height,
    seed: generation.seed.seed,
    img2img: generation.img2img.isEnabled
      ? new Img2ImgRequest().load({
          source: generation.img2img.source,
          noise: generation.img2img.noise,
        })
      : null,
    lora: generation.lora.isEnabled
      ? new LoraRequest().load({ entries: generation.lora.entries.map(stateToLoraEntry) })
      : null,
    control_net: generation.controlNet.isEnabled
      ? new ControlNetRequest().load({
          conditions: generation.controlNet.conditions.map(stateToControlNetConditionRequest),
        })
      : null,
    refiner: generation.refiner.isEnabled
      ? new RefinerRequest().load(
          generation.refiner.ensembleMode
            ? {
                cfg_scale: generation.refiner.cfgScale,
                high_noise_end: generation.refiner.highNoiseEnd,
              }
            : {
                cfg_scale: generation.refiner.cfgScale,
                steps: generation.refiner.steps,
                noise: generation.refiner.noise,
              }
        )
      : null,
    upscale: generation.upscale.isEnabled
      ? new UpscaleRequest().load({
          factor: generation.upscale.factor,
          denoising: generation.upscale.denoising,
          blend: generation.upscale.blend,
        })
      : null,
    face: generation.face.isEnabled
      ? new FaceRestorationRequest().load({
          blend: generation.face.blend,
        })
      : null,
    high_res: generation.highRes.isEnabled
      ? new HighResRequest().load({
          factor: generation.highRes.factor,
          steps: generation.highRes.steps,
          cfg_scale: generation.highRes.cfgScale,
          noise: generation.highRes.noise,
        })
      : null,
    inpaint: generation.inpaint.isEnabled
      ? new InpaintRequest().load({
          source: generation.inpaint.source,
          use_alpha_channel: generation.inpaint.useAlphaChannel,
          invert_mask: generation.inpaint.invertMask,
        })
      : null,
  });
  return req;
}

export function updateAll(generation: GenerationParamsState, req: ImageRequest) {
  updatePrompt(generation, req);
  updateGeneral(generation, req);
  updateSeed(generation, req);
  updateImg2Img(generation, req);
  updateLora(generation, req);
  updateControlNet(generation, req);
  updatePostProcessing(generation, req);
  updateInpaint(generation, req);
}

export function updatePrompt(generation: GenerationParamsState, req: ImageRequest) {
  generation.prompt.prompt = req.prompt;
  generation.prompt.negativePrompt = req.negative_prompt;
}

export function updateGeneral(generation: GenerationParamsState, req: ImageRequest) {
  generation.general.model = req.model;
  generation.general.scheduler = req.scheduler;
  generation.general.steps = req.steps;
  generation.general.cfgScale = req.cfg_scale;
  generation.general.width = req.width;
  generation.general.height = req.height;
}

export function updateSeed(generation: GenerationParamsState, req: ImageRequest) {
  generation.seed.isEnabled = true;
  generation.seed.seed = req.seed;
}

export function updateSourceImages(generation: GenerationParamsState, req: ImageRequest) {
  if (req.img2img) {
    generation.img2img.source = req.img2img.source;
  }

  if (req.control_net) {
    while (generation.controlNet.conditions.length < req.control_net.conditions.length) {
      generation.controlNet.conditions.push(new ControlNetConditionParamsState());
    }

    req.control_net.conditions.map((conditionMeta, index) => {
      const conditionParam = generation.controlNet.conditions[index];
      conditionParam.source = conditionMeta.source;
    });
  }
}

export function updateImg2Img(generation: GenerationParamsState, req: ImageRequest) {
  if (req.img2img) {
    generation.img2img.isEnabled = true;
    generation.img2img.source = req.img2img.source;
    generation.img2img.noise = req.img2img.noise;
  } else {
    generation.img2img.isEnabled = false;
    generation.img2img.source = "";
  }
}

export function updateLora(generation: GenerationParamsState, req: ImageRequest) {
  if (req.lora) {
    generation.lora.isEnabled = true;

    while (generation.lora.entries.length > req.lora.entries.length) {
      generation.lora.entries.pop();
    }

    while (generation.lora.entries.length < req.lora.entries.length) {
      generation.lora.entries.push(new LoraModelParamsState());
    }

    req.lora.entries.map((loraModelReq, index) => {
      const loraModelParamsState = generation.lora.entries[index];
      loraModelParamsState.model = loraModelReq.model;
      loraModelParamsState.weight = loraModelReq.weight;
    });
  } else {
    generation.lora.isEnabled = false;
  }
}

export function updateControlNet(generation: GenerationParamsState, req: ImageRequest) {
  if (req.control_net) {
    generation.controlNet.isEnabled = true;

    while (generation.controlNet.conditions.length > req.control_net.conditions.length) {
      generation.controlNet.conditions.pop();
    }

    while (generation.controlNet.conditions.length < req.control_net.conditions.length) {
      generation.controlNet.conditions.push(new ControlNetConditionParamsState());
    }

    req.control_net.conditions.map((conditionMeta, index) => {
      const conditionParam = generation.controlNet.conditions[index];
      conditionParam.model = conditionMeta.model;
      conditionParam.source = conditionMeta.source;
      conditionParam.processor = conditionMeta.processor;
      conditionParam.params = conditionMeta.params;
      conditionParam.guidanceStart = conditionMeta.guidance_start;
      conditionParam.guidanceEnd = conditionMeta.guidance_end;
      conditionParam.scale = conditionMeta.scale;
    });
  } else {
    generation.controlNet.isEnabled = false;
    generation.controlNet.conditions.length = 0;
  }
}

export function updatePostProcessing(generation: GenerationParamsState, req: ImageRequest) {
  if (req.refiner) {
    generation.refiner.isEnabled = true;
    generation.refiner.ensembleMode = req.refiner.high_noise_end !== undefined;
    generation.refiner.cfgScale = req.refiner.cfg_scale;
    if (req.refiner.high_noise_end !== undefined) generation.refiner.highNoiseEnd = req.refiner.high_noise_end;
    if (req.refiner.steps !== undefined) generation.refiner.steps = req.refiner.steps;
    if (req.refiner.noise !== undefined) generation.refiner.noise = req.refiner.noise;
  } else {
    generation.refiner.isEnabled = false;
  }

  if (req.upscale) {
    generation.upscale.isEnabled = true;
    generation.upscale.factor = req.upscale.factor;
    generation.upscale.denoising = req.upscale.denoising;
    generation.upscale.blend = req.upscale.blend;
  } else {
    generation.upscale.isEnabled = false;
  }

  if (req.face) {
    generation.face.isEnabled = true;
    generation.face.blend = req.face.blend;
  } else {
    generation.face.isEnabled = false;
  }

  if (req.high_res) {
    generation.highRes.isEnabled = true;
    generation.highRes.factor = req.high_res.factor;
    generation.highRes.steps = req.high_res.steps;
    generation.highRes.cfgScale = req.high_res.cfg_scale;
    generation.highRes.noise = req.high_res.noise;
  } else {
    generation.highRes.isEnabled = false;
  }
}

export function updateInpaint(generation: GenerationParamsState, req: ImageRequest) {
  if (req.inpaint) {
    generation.inpaint.isEnabled = true;
    generation.inpaint.source = req.inpaint.source;
    generation.inpaint.useAlphaChannel = req.inpaint.use_alpha_channel;
    generation.inpaint.invertMask = req.inpaint.invert_mask;
  } else {
    generation.inpaint.isEnabled = false;
  }
}
