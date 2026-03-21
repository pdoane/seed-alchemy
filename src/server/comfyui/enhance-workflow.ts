// ComfyUI enhance workflow generation

import type { Architecture } from "../../shared/types/models.js";
import type { EnhanceParams } from "../../shared/types/image.js";
import type { ComfyWorkflow } from "./types.js";

// Generate an enhance workflow for face detailer or upscale operations
export function generateEnhanceWorkflow(
  params: EnhanceParams,
  sourceImage: string,
  architecture?: Architecture
): ComfyWorkflow {
  const isZit = architecture === "zit";
  const workflow: ComfyWorkflow = {};

  // Load source image
  workflow["source"] = {
    class_type: "LoadImage",
    inputs: {
      image: sourceImage,
    },
  };

  let imageRef: [string, number] = ["source", 0];

  if (params.enhanceType === "face_detailer") {
    // Face Detailer requires full model pipeline (it does inpainting)
    let modelRef: [string, number];
    let clipRef: [string, number];
    let vaeRef: [string, number];

    // Model loading
    if (isZit) {
      workflow["unet"] = {
        class_type: "UNETLoader",
        inputs: {
          unet_name: params.model,
          weight_dtype: "default",
        },
      };
      workflow["clip"] = {
        class_type: "CLIPLoader",
        inputs: {
          clip_name: "qwen_3_4b.safetensors",
          type: "lumina2",
          device: "default",
        },
      };
      workflow["vae"] = {
        class_type: "VAELoader",
        inputs: {
          vae_name: "ae.safetensors",
        },
      };
      modelRef = ["unet", 0];
      clipRef = ["clip", 0];
      vaeRef = ["vae", 0];
    } else {
      workflow["checkpoint"] = {
        class_type: "CheckpointLoaderSimple",
        inputs: {
          ckpt_name: params.model,
        },
      };
      modelRef = ["checkpoint", 0];
      clipRef = ["checkpoint", 1];
      vaeRef = ["checkpoint", 2];
    }

    // LoRA chain
    const loras = params.loras || [];
    for (const [i, lora] of loras.entries()) {
      const nodeId = `lora_${i}`;
      workflow[nodeId] = {
        class_type: "LoraLoader",
        inputs: {
          lora_name: lora.filename,
          strength_model: lora.weight,
          strength_clip: lora.weight,
          model: modelRef,
          clip: clipRef,
        },
      };
      modelRef = [nodeId, 0];
      clipRef = [nodeId, 1];
    }

    // Prompt encoding
    workflow["positive"] = {
      class_type: "CLIPTextEncode",
      inputs: {
        text: params.prompt,
        clip: clipRef,
      },
    };

    if (params.negativePrompt.trim()) {
      workflow["negative"] = {
        class_type: "CLIPTextEncode",
        inputs: {
          text: params.negativePrompt,
          clip: clipRef,
        },
      };
    } else {
      workflow["negative"] = {
        class_type: "ConditioningZeroOut",
        inputs: {
          conditioning: ["positive", 0],
        },
      };
    }

    const positiveRef: [string, number] = ["positive", 0];
    const negativeRef: [string, number] = ["negative", 0];

    // Face detector
    workflow["bbox_detector"] = {
      class_type: "UltralyticsDetectorProvider",
      inputs: {
        model_name: "bbox/face_yolov8m.pt",
      },
    };

    // FaceDetailer node
    workflow["face_detailer"] = {
      class_type: "FaceDetailer",
      inputs: {
        image: imageRef,
        model: modelRef,
        clip: clipRef,
        vae: vaeRef,
        positive: positiveRef,
        negative: negativeRef,
        bbox_detector: ["bbox_detector", 0],
        guide_size: 512,
        guide_size_for: "bbox",
        max_size: 1024,
        seed: params.seed,
        steps: params.steps,
        cfg: params.cfgScale,
        sampler_name: params.sampler,
        scheduler: params.scheduler,
        denoise: 0.5,
        feather: 5,
        noise_mask: true,
        force_inpaint: true,
        bbox_threshold: 0.5,
        bbox_dilation: 10,
        bbox_crop_factor: 3,
        sam_detection_hint: "center-1",
        sam_dilation: 0,
        sam_threshold: 0.93,
        sam_bbox_expansion: 0,
        sam_mask_hint_threshold: 0.7,
        sam_mask_hint_use_negative: "False",
        drop_size: 10,
        wildcard: "",
        cycle: 1,
      },
    };

    imageRef = ["face_detailer", 0];
  } else if (params.enhanceType === "upscale") {
    // Workaround for ComfyUI tensor stride issue with LoadImage
    // See: https://github.com/comfyanonymous/ComfyUI/issues/5075
    workflow["fix_stride"] = {
      class_type: "ImageScaleBy",
      inputs: {
        image: imageRef,
        upscale_method: "lanczos",
        scale_by: 1.0,
      },
    };
    imageRef = ["fix_stride", 0];

    // Load upscale model
    workflow["upscaler_model"] = {
      class_type: "UpscaleModelLoader",
      inputs: {
        model_name: "RealESRGAN_x4.pth",
      },
    };

    workflow["upscale"] = {
      class_type: "ImageUpscaleWithModel",
      inputs: {
        upscale_model: ["upscaler_model", 0],
        image: imageRef,
      },
    };

    imageRef = ["upscale", 0];

    // Rescale to target factor (assumes 4x upscale model)
    const modelScale = 4;
    const targetScale = params.upscaleFactor || 2;
    if (targetScale !== modelScale) {
      const rescaleFactor = targetScale / modelScale;
      workflow["rescale"] = {
        class_type: "ImageScaleBy",
        inputs: {
          image: imageRef,
          upscale_method: "lanczos",
          scale_by: rescaleFactor,
        },
      };
      imageRef = ["rescale", 0];
    }
  }

  // Save
  workflow["save"] = {
    class_type: "SaveImage",
    inputs: {
      images: imageRef,
      filename_prefix: "SeedAlchemy",
    },
  };

  return workflow;
}
