// ComfyUI workflow generation

import type { Architecture } from "../../shared/types/models.js";
import type { ImageParams } from "../../shared/types/image.js";
import type { ControlNetType } from "../../shared/types/controlnet.js";
import type { ComfyWorkflow } from "./types.js";

// Check if architecture supports IP Adapter
function supportsIPAdapter(architecture?: Architecture): boolean {
  return architecture === "sd15" || architecture === "sdxl";
}

// Check if architecture supports ControlNet
function supportsControlNet(architecture?: Architecture): boolean {
  return architecture === "sdxl" || architecture === "sd15";
}

// Get Union ControlNet model path (SDXL)
function getUnionControlNetFile(): string {
  return "SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model.safetensors";
}

// SD1.5 ControlNet 1.1 model filenames
const SD15_CONTROLNET_FILES: Partial<Record<ControlNetType, string>> = {
  canny: "control_v11p_sd15_canny_fp16.safetensors",
  openpose: "control_v11p_sd15_openpose_fp16.safetensors",
  depth: "control_v11f1p_sd15_depth_fp16.safetensors",
  normal: "control_v11p_sd15_normalbae_fp16.safetensors",
  lineart: "control_v11p_sd15_lineart_fp16.safetensors",
  lineart_anime: "control_v11p_sd15s2_lineart_anime_fp16.safetensors",
  scribble: "control_v11p_sd15_scribble_fp16.safetensors",
  softedge: "control_v11p_sd15_softedge_fp16.safetensors",
  segment: "control_v11p_sd15_seg_fp16.safetensors",
  tile: "control_v11f1e_sd15_tile_fp16.safetensors",
  mlsd: "control_v11p_sd15_mlsd_fp16.safetensors",
  shuffle: "control_v11e_sd15_shuffle_fp16.safetensors",
  inpaint: "control_v11p_sd15_inpaint_fp16.safetensors",
};

// Get SD1.5 ControlNet model path
function getSD15ControlNetFile(type: ControlNetType): string {
  const filename = SD15_CONTROLNET_FILES[type];
  if (!filename) {
    throw new Error(`No SD1.5 ControlNet model for type: ${type}`);
  }
  return `1.5/${filename}`;
}

// Map ControlNet type to SetUnionControlNetType string value (SDXL only)
const SDXL_CONTROLNET_TYPE_VALUES: Partial<Record<ControlNetType, string>> = {
  openpose: "openpose",
  depth: "depth",
  scribble: "hed/pidi/scribble/ted",
  lineart: "canny/lineart/anime_lineart/mlsd",
  normal: "normal",
  segment: "segment",
  tile: "tile",
  repaint: "repaint",
};

// Get CLIP Vision model filename
function getClipVisionFile(): string {
  return "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors";
}

// Get IP Adapter model filename (always uses "plus" variant for better quality)
function getIPAdapterFile(architecture: Architecture): string {
  if (architecture === "sdxl") {
    return "ip-adapter-plus_sdxl_vit-h.safetensors";
  }
  return "ip-adapter-plus_sd15.safetensors";
}

// Generate an image workflow
// Note: seed should be resolved before calling this function (seed=-1 should be replaced with actual random seed)
export function generateImgWorkflow(
  params: ImageParams,
  imageCount: number = 1,
  architecture?: Architecture
): ComfyWorkflow {
  const isZit = architecture === "zit";
  const workflow: ComfyWorkflow = {};

  // Track references throughout the workflow
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

  // ZIT model sampling
  if (isZit) {
    workflow["model_sampling"] = {
      class_type: "ModelSamplingAuraFlow",
      inputs: {
        model: modelRef,
        shift: 3,
      },
    };
    modelRef = ["model_sampling", 0];
  }

  // IP Adapter
  const referenceImages = params.referenceImages || [];
  if (referenceImages.length > 0 && supportsIPAdapter(architecture)) {
    // Load CLIP Vision model
    workflow["clip_vision_loader"] = {
      class_type: "CLIPVisionLoader",
      inputs: {
        clip_name: getClipVisionFile(),
      },
    };

    // Load IP Adapter model
    workflow["ipadapter_loader"] = {
      class_type: "IPAdapterModelLoader",
      inputs: {
        ipadapter_file: getIPAdapterFile(architecture!),
      },
    };

    // Load all reference images
    for (const [i, refImg] of referenceImages.entries()) {
      workflow[`ref_image_${i}`] = {
        class_type: "LoadImage",
        inputs: {
          image: refImg.filename,
        },
      };
    }

    // Batch images if more than one
    let imageRef: [string, number] = ["ref_image_0", 0];
    for (let i = 1; i < referenceImages.length; i++) {
      const batchNodeId = `image_batch_${i}`;
      workflow[batchNodeId] = {
        class_type: "ImageBatch",
        inputs: {
          image1: imageRef,
          image2: [`ref_image_${i}`, 0],
        },
      };
      imageRef = [batchNodeId, 0];
    }

    // Apply IPAdapterAdvanced
    workflow["ipadapter"] = {
      class_type: "IPAdapterAdvanced",
      inputs: {
        model: modelRef,
        ipadapter: ["ipadapter_loader", 0],
        image: imageRef,
        weight: params.referenceWeight,
        weight_type: params.referenceWeightType,
        combine_embeds: params.referenceCombineMode,
        start_at: 0,
        end_at: 1,
        embeds_scaling: "V only",
        clip_vision: ["clip_vision_loader", 0],
      },
    };
    modelRef = ["ipadapter", 0];
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
    // Use ConditioningZeroOut to ensure same tensor shape as positive
    workflow["negative"] = {
      class_type: "ConditioningZeroOut",
      inputs: {
        conditioning: ["positive", 0],
      },
    };
  }

  // Track conditioning references (may be modified by ControlNet)
  let positiveRef: [string, number] = ["positive", 0];
  let negativeRef: [string, number] = ["negative", 0];

  // ControlNet chain
  const controlNets = params.controlNets || [];
  if (controlNets.length > 0 && supportsControlNet(architecture)) {
    if (architecture === "sdxl") {
      // SDXL: Load Union ControlNet model once, set type per use
      workflow["controlnet_loader"] = {
        class_type: "ControlNetLoader",
        inputs: {
          control_net_name: getUnionControlNetFile(),
        },
      };

      for (const [i, config] of controlNets.entries()) {
        // Load control image
        workflow[`ctrl_image_${i}`] = {
          class_type: "LoadImage",
          inputs: {
            image: config.image,
          },
        };

        let ctrlImageRef: [string, number] = [`ctrl_image_${i}`, 0];

        // Optional: preprocess image
        if (config.preprocessor) {
          workflow[`ctrl_preprocess_${i}`] = {
            class_type: "AIO_Preprocessor",
            inputs: {
              image: ctrlImageRef,
              preprocessor: config.preprocessor,
              resolution: Math.min(params.width, params.height),
            },
          };
          ctrlImageRef = [`ctrl_preprocess_${i}`, 0];
        }

        // Set union control type
        const typeValue = SDXL_CONTROLNET_TYPE_VALUES[config.type];
        if (!typeValue) {
          continue; // Skip unsupported types
        }
        workflow[`ctrl_type_${i}`] = {
          class_type: "SetUnionControlNetType",
          inputs: {
            control_net: ["controlnet_loader", 0],
            type: typeValue,
          },
        };

        // Apply ControlNet to conditioning
        workflow[`ctrl_apply_${i}`] = {
          class_type: "ControlNetApplyAdvanced",
          inputs: {
            positive: positiveRef,
            negative: negativeRef,
            control_net: [`ctrl_type_${i}`, 0],
            image: ctrlImageRef,
            strength: config.weight,
            start_percent: 0,
            end_percent: 1,
          },
        };

        positiveRef = [`ctrl_apply_${i}`, 0];
        negativeRef = [`ctrl_apply_${i}`, 1];
      }
    } else if (architecture === "sd15") {
      // SD1.5: Load separate ControlNet model per ControlNet
      for (const [i, config] of controlNets.entries()) {
        // Load ControlNet model for this type
        workflow[`controlnet_loader_${i}`] = {
          class_type: "ControlNetLoader",
          inputs: {
            control_net_name: getSD15ControlNetFile(config.type),
          },
        };

        // Load control image
        workflow[`ctrl_image_${i}`] = {
          class_type: "LoadImage",
          inputs: {
            image: config.image,
          },
        };

        let ctrlImageRef: [string, number] = [`ctrl_image_${i}`, 0];

        // Optional: preprocess image
        if (config.preprocessor) {
          workflow[`ctrl_preprocess_${i}`] = {
            class_type: "AIO_Preprocessor",
            inputs: {
              image: ctrlImageRef,
              preprocessor: config.preprocessor,
              resolution: Math.min(params.width, params.height),
            },
          };
          ctrlImageRef = [`ctrl_preprocess_${i}`, 0];
        }

        // Apply ControlNet to conditioning (no type setter needed)
        workflow[`ctrl_apply_${i}`] = {
          class_type: "ControlNetApplyAdvanced",
          inputs: {
            positive: positiveRef,
            negative: negativeRef,
            control_net: [`controlnet_loader_${i}`, 0],
            image: ctrlImageRef,
            strength: config.weight,
            start_percent: 0,
            end_percent: 1,
          },
        };

        positiveRef = [`ctrl_apply_${i}`, 0];
        negativeRef = [`ctrl_apply_${i}`, 1];
      }
    }
  }

  // Latent generation - either from source image or empty
  let latentRef: [string, number];
  let denoise = 1;

  if (params.sourceImage) {
    // Load source image
    workflow["source_image"] = {
      class_type: "LoadImage",
      inputs: {
        image: params.sourceImage,
      },
    };

    // Resize source image to target dimensions
    workflow["source_resize"] = {
      class_type: "ResizeAndPadImage",
      inputs: {
        image: ["source_image", 0],
        target_width: params.width,
        target_height: params.height,
        padding_color: "black",
        interpolation: "lanczos",
      },
    };

    // Encode source image to latent
    workflow["source_encode"] = {
      class_type: "VAEEncode",
      inputs: {
        pixels: ["source_resize", 0],
        vae: vaeRef,
      },
    };

    latentRef = ["source_encode", 0];
    denoise = params.sourceImageStrength;
  } else {
    // Generate empty latent
    workflow["latent"] = {
      class_type: isZit ? "EmptySD3LatentImage" : "EmptyLatentImage",
      inputs: {
        width: params.width,
        height: params.height,
        batch_size: imageCount,
      },
    };
    latentRef = ["latent", 0];
  }

  // Sampler
  const samplerName = params.sampler;
  const scheduler = params.scheduler;

  workflow["sampler"] = {
    class_type: "KSampler",
    inputs: {
      model: modelRef,
      positive: positiveRef,
      negative: negativeRef,
      latent_image: latentRef,
      seed: params.seed,
      steps: params.steps,
      cfg: params.cfgScale,
      sampler_name: samplerName,
      scheduler: scheduler,
      denoise: denoise,
    },
  };

  // VAE Decode
  workflow["decode"] = {
    class_type: "VAEDecode",
    inputs: {
      samples: ["sampler", 0],
      vae: vaeRef,
    },
  };

  // Track the final image output - may be updated by FaceDetailer/Upscaler
  let imageRef: [string, number] = ["decode", 0];

  // FaceDetailer
  if (params.faceDetailer) {
    workflow["bbox_detector"] = {
      class_type: "UltralyticsDetectorProvider",
      inputs: {
        model_name: "bbox/face_yolov8m.pt",
      },
    };

    workflow["face_detailer"] = {
      class_type: "FaceDetailer",
      inputs: {
        image: ["decode", 0],
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
        steps: 10,
        cfg: params.cfgScale,
        sampler_name: samplerName,
        scheduler: "simple",
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
  }

  // Upscaler
  if (params.upscaleEnabled) {
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
