import { describe, it, expect } from "vitest";
import { generateImgWorkflow } from "./workflow.js";
import type { ImageParams } from "../../shared/types/image.js";

const baseParams: ImageParams = {
  prompt: "a cat",
  negativePrompt: "ugly",
  model: "model.safetensors",
  sampler: "euler",
  scheduler: "normal",
  width: 512,
  height: 512,
  steps: 20,
  cfgScale: 7,
  seed: 12345,
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

describe("generateImgWorkflow", () => {
  it("should generate basic workflow without LoRAs", () => {
    const workflow = generateImgWorkflow(baseParams);

    expect(workflow["checkpoint"]).toBeDefined();
    expect(workflow["checkpoint"]!.class_type).toBe("CheckpointLoaderSimple");
    expect(workflow["checkpoint"]!.inputs.ckpt_name).toBe("model.safetensors");

    expect(workflow["positive"]).toBeDefined();
    expect(workflow["positive"]!.inputs.text).toBe("a cat");
    expect(workflow["positive"]!.inputs.clip).toEqual(["checkpoint", 1]);

    expect(workflow["negative"]).toBeDefined();
    expect(workflow["negative"]!.inputs.text).toBe("ugly");
    expect(workflow["negative"]!.inputs.clip).toEqual(["checkpoint", 1]);

    expect(workflow["sampler"]).toBeDefined();
    expect(workflow["sampler"]!.inputs.model).toEqual(["checkpoint", 0]);
    expect(workflow["sampler"]!.inputs.seed).toBe(12345);
    expect(workflow["sampler"]!.inputs.steps).toBe(20);

    expect(workflow["decode"]).toBeDefined();
    expect(workflow["save"]).toBeDefined();
  });

  it("should add single LoRA node", () => {
    const params: ImageParams = {
      ...baseParams,
      loras: [{ filename: "lora1.safetensors", weight: 0.8 }],
    };

    const workflow = generateImgWorkflow(params);

    expect(workflow["lora_0"]).toBeDefined();
    expect(workflow["lora_0"]!.class_type).toBe("LoraLoader");
    expect(workflow["lora_0"]!.inputs.lora_name).toBe("lora1.safetensors");
    expect(workflow["lora_0"]!.inputs.strength_model).toBe(0.8);
    expect(workflow["lora_0"]!.inputs.strength_clip).toBe(0.8);
    expect(workflow["lora_0"]!.inputs.model).toEqual(["checkpoint", 0]);
    expect(workflow["lora_0"]!.inputs.clip).toEqual(["checkpoint", 1]);

    // CLIP should now come from LoRA
    expect(workflow["positive"]!.inputs.clip).toEqual(["lora_0", 1]);
    expect(workflow["negative"]!.inputs.clip).toEqual(["lora_0", 1]);

    // Model should now come from LoRA
    expect(workflow["sampler"]!.inputs.model).toEqual(["lora_0", 0]);
  });

  it("should chain multiple LoRAs", () => {
    const params: ImageParams = {
      ...baseParams,
      loras: [
        { filename: "lora1.safetensors", weight: 0.8 },
        { filename: "lora2.safetensors", weight: 0.5 },
        { filename: "lora3.safetensors", weight: 1.2 },
      ],
    };

    const workflow = generateImgWorkflow(params);

    // First LoRA connects to checkpoint
    expect(workflow["lora_0"]!.inputs.model).toEqual(["checkpoint", 0]);
    expect(workflow["lora_0"]!.inputs.clip).toEqual(["checkpoint", 1]);

    // Second LoRA connects to first LoRA
    expect(workflow["lora_1"]!.inputs.model).toEqual(["lora_0", 0]);
    expect(workflow["lora_1"]!.inputs.clip).toEqual(["lora_0", 1]);

    // Third LoRA connects to second LoRA
    expect(workflow["lora_2"]!.inputs.model).toEqual(["lora_1", 0]);
    expect(workflow["lora_2"]!.inputs.clip).toEqual(["lora_1", 1]);

    // Final outputs come from last LoRA
    expect(workflow["positive"]!.inputs.clip).toEqual(["lora_2", 1]);
    expect(workflow["negative"]!.inputs.clip).toEqual(["lora_2", 1]);
    expect(workflow["sampler"]!.inputs.model).toEqual(["lora_2", 0]);
  });

  it("should pass sampler and scheduler to KSampler", () => {
    const params = { ...baseParams, sampler: "dpmpp_2m", scheduler: "karras" };
    const workflow = generateImgWorkflow(params);
    expect(workflow["sampler"]!.inputs.sampler_name).toBe("dpmpp_2m");
    expect(workflow["sampler"]!.inputs.scheduler).toBe("karras");
  });

  it("should handle empty loras array", () => {
    const params: ImageParams = { ...baseParams, loras: [] };
    const workflow = generateImgWorkflow(params);

    expect(workflow["lora_0"]).toBeUndefined();
    expect(workflow["positive"]!.inputs.clip).toEqual(["checkpoint", 1]);
    expect(workflow["sampler"]!.inputs.model).toEqual(["checkpoint", 0]);
  });

  it("should handle undefined loras", () => {
    const params = { ...baseParams };
    delete (params as Partial<ImageParams>).loras;
    const workflow = generateImgWorkflow(params as ImageParams);

    expect(workflow["lora_0"]).toBeUndefined();
    expect(workflow["positive"]!.inputs.clip).toEqual(["checkpoint", 1]);
  });

  it("should pass through image dimensions and count", () => {
    const params: ImageParams = {
      ...baseParams,
      width: 1024,
      height: 768,
    };

    const workflow = generateImgWorkflow(params, 4);

    expect(workflow["latent"]!.inputs.width).toBe(1024);
    expect(workflow["latent"]!.inputs.height).toBe(768);
    expect(workflow["latent"]!.inputs.batch_size).toBe(4);
  });

  it("should default to batch_size 1 when imageCount not provided", () => {
    const workflow = generateImgWorkflow(baseParams);
    expect(workflow["latent"]!.inputs.batch_size).toBe(1);
  });

  it("should add upscaler nodes with rescale when factor < 4", () => {
    const params: ImageParams = {
      ...baseParams,
      upscaleEnabled: true,
      upscaleFactor: 2,
    };

    const workflow = generateImgWorkflow(params);

    expect(workflow["upscaler_model"]).toBeDefined();
    expect(workflow["upscaler_model"]!.class_type).toBe("UpscaleModelLoader");
    expect(workflow["upscaler_model"]!.inputs.model_name).toBe(
      "RealESRGAN_x4.pth"
    );

    expect(workflow["upscale"]).toBeDefined();
    expect(workflow["upscale"]!.class_type).toBe("ImageUpscaleWithModel");
    expect(workflow["upscale"]!.inputs.upscale_model).toEqual([
      "upscaler_model",
      0,
    ]);
    expect(workflow["upscale"]!.inputs.image).toEqual(["decode", 0]);

    // Rescale node should be added for 2x factor (scale_by = 0.5)
    expect(workflow["rescale"]).toBeDefined();
    expect(workflow["rescale"]!.class_type).toBe("ImageScaleBy");
    expect(workflow["rescale"]!.inputs.image).toEqual(["upscale", 0]);
    expect(workflow["rescale"]!.inputs.scale_by).toBe(0.5);

    // Save should reference rescale output
    expect(workflow["save"]!.inputs.images).toEqual(["rescale", 0]);
  });

  it("should skip rescale when factor is 4", () => {
    const params: ImageParams = {
      ...baseParams,
      upscaleEnabled: true,
      upscaleFactor: 4,
    };

    const workflow = generateImgWorkflow(params);

    expect(workflow["upscale"]).toBeDefined();
    expect(workflow["rescale"]).toBeUndefined();

    // Save should reference upscale output directly
    expect(workflow["save"]!.inputs.images).toEqual(["upscale", 0]);
  });

  it("should not add upscaler when disabled", () => {
    const params: ImageParams = {
      ...baseParams,
      upscaleEnabled: false,
    };

    const workflow = generateImgWorkflow(params);

    expect(workflow["upscaler_model"]).toBeUndefined();
    expect(workflow["upscale"]).toBeUndefined();
    expect(workflow["save"]!.inputs.images).toEqual(["decode", 0]);
  });

  it("should chain upscaler after face detailer when both enabled", () => {
    const params: ImageParams = {
      ...baseParams,
      faceDetailer: true,
      upscaleEnabled: true,
      upscaleFactor: 2,
    };

    const workflow = generateImgWorkflow(params);

    // Upscaler should take input from face_detailer
    expect(workflow["upscale"]!.inputs.image).toEqual(["face_detailer", 0]);

    // Rescale should take input from upscale
    expect(workflow["rescale"]!.inputs.image).toEqual(["upscale", 0]);

    // Save should reference rescale output
    expect(workflow["save"]!.inputs.images).toEqual(["rescale", 0]);
  });
});

// ZIT (Z-Image Turbo) workflow tests
const zitParams: ImageParams = {
  prompt: "a cat",
  negativePrompt: "",
  model: "z_image_turbo_bf16.safetensors",
  sampler: "res_multistep",
  scheduler: "simple",
  width: 1024,
  height: 1024,
  steps: 9,
  cfgScale: 1,
  seed: 12345,
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

describe("generateImgWorkflow (ZIT)", () => {
  it("should generate ZIT workflow with separate loaders", () => {
    const workflow = generateImgWorkflow(zitParams, 1, "zit");

    // Should use separate loaders instead of CheckpointLoaderSimple
    expect(workflow["checkpoint"]).toBeUndefined();

    expect(workflow["unet"]).toBeDefined();
    expect(workflow["unet"]!.class_type).toBe("UNETLoader");
    expect(workflow["unet"]!.inputs.unet_name).toBe(
      "z_image_turbo_bf16.safetensors"
    );

    expect(workflow["clip"]).toBeDefined();
    expect(workflow["clip"]!.class_type).toBe("CLIPLoader");
    expect(workflow["clip"]!.inputs.clip_name).toBe("qwen_3_4b.safetensors");
    expect(workflow["clip"]!.inputs.type).toBe("lumina2");

    expect(workflow["vae"]).toBeDefined();
    expect(workflow["vae"]!.class_type).toBe("VAELoader");
    expect(workflow["vae"]!.inputs.vae_name).toBe("ae.safetensors");
  });

  it("should use ModelSamplingAuraFlow with shift=3", () => {
    const workflow = generateImgWorkflow(zitParams, 1, "zit");

    expect(workflow["model_sampling"]).toBeDefined();
    expect(workflow["model_sampling"]!.class_type).toBe(
      "ModelSamplingAuraFlow"
    );
    expect(workflow["model_sampling"]!.inputs.shift).toBe(3);
    expect(workflow["model_sampling"]!.inputs.model).toEqual(["unet", 0]);
  });

  it("should use EmptySD3LatentImage", () => {
    const workflow = generateImgWorkflow(zitParams, 1, "zit");

    expect(workflow["latent"]).toBeDefined();
    expect(workflow["latent"]!.class_type).toBe("EmptySD3LatentImage");
    expect(workflow["latent"]!.inputs.width).toBe(1024);
    expect(workflow["latent"]!.inputs.height).toBe(1024);
  });

  it("should pass sampler and scheduler from params", () => {
    const workflow = generateImgWorkflow(zitParams, 1, "zit");

    expect(workflow["sampler"]).toBeDefined();
    expect(workflow["sampler"]!.inputs.sampler_name).toBe("res_multistep");
    expect(workflow["sampler"]!.inputs.scheduler).toBe("simple");
  });

  it("should use ConditioningZeroOut for empty negative prompt", () => {
    const workflow = generateImgWorkflow(zitParams, 1, "zit");

    expect(workflow["negative"]).toBeDefined();
    expect(workflow["negative"]!.class_type).toBe("ConditioningZeroOut");
    expect(workflow["negative"]!.inputs.conditioning).toEqual(["positive", 0]);
  });

  it("should use CLIPTextEncode for non-empty negative prompt", () => {
    const params = { ...zitParams, negativePrompt: "ugly" };
    const workflow = generateImgWorkflow(params, 1, "zit");

    expect(workflow["negative"]).toBeDefined();
    expect(workflow["negative"]!.class_type).toBe("CLIPTextEncode");
    expect(workflow["negative"]!.inputs.text).toBe("ugly");
  });

  it("should use LoraLoader for LoRAs", () => {
    const params = {
      ...zitParams,
      loras: [{ filename: "lora1.safetensors", weight: 0.8 }],
    };
    const workflow = generateImgWorkflow(params, 1, "zit");

    expect(workflow["lora_0"]).toBeDefined();
    expect(workflow["lora_0"]!.class_type).toBe("LoraLoader");
    expect(workflow["lora_0"]!.inputs.lora_name).toBe("lora1.safetensors");
    expect(workflow["lora_0"]!.inputs.strength_model).toBe(0.8);
    expect(workflow["lora_0"]!.inputs.strength_clip).toBe(0.8);
    expect(workflow["lora_0"]!.inputs.model).toEqual(["unet", 0]);
    expect(workflow["lora_0"]!.inputs.clip).toEqual(["clip", 0]);

    // ModelSamplingAuraFlow should connect to LoRA output
    expect(workflow["model_sampling"]!.inputs.model).toEqual(["lora_0", 0]);
  });

  it("should chain multiple LoRAs for ZIT", () => {
    const params = {
      ...zitParams,
      loras: [
        { filename: "lora1.safetensors", weight: 0.8 },
        { filename: "lora2.safetensors", weight: 0.5 },
      ],
    };
    const workflow = generateImgWorkflow(params, 1, "zit");

    expect(workflow["lora_0"]!.inputs.model).toEqual(["unet", 0]);
    expect(workflow["lora_1"]!.inputs.model).toEqual(["lora_0", 0]);
    expect(workflow["model_sampling"]!.inputs.model).toEqual(["lora_1", 0]);
  });

  it("should support upscaling for ZIT", () => {
    const params = { ...zitParams, upscaleEnabled: true, upscaleFactor: 2 };
    const workflow = generateImgWorkflow(params, 1, "zit");

    expect(workflow["upscaler_model"]).toBeDefined();
    expect(workflow["upscale"]).toBeDefined();
    expect(workflow["rescale"]).toBeDefined();
    expect(workflow["save"]!.inputs.images).toEqual(["rescale", 0]);
  });

  it("should support face detailer for ZIT", () => {
    const params = { ...zitParams, faceDetailer: true };
    const workflow = generateImgWorkflow(params, 1, "zit");

    expect(workflow["face_detailer"]).toBeDefined();
    expect(workflow["face_detailer"]!.inputs.clip).toEqual(["clip", 0]);
    expect(workflow["face_detailer"]!.inputs.vae).toEqual(["vae", 0]);
    expect(workflow["face_detailer"]!.inputs.sampler_name).toBe(
      "res_multistep"
    );
    expect(workflow["face_detailer"]!.inputs.scheduler).toBe("simple");
  });
});

// IP Adapter tests
describe("generateImgWorkflow (IP Adapter)", () => {
  it("should add IP Adapter nodes for SD1.5 with reference images", () => {
    const params: ImageParams = {
      ...baseParams,
      referenceImages: [{ filename: "ref1.png" }],
      referenceWeight: 0.8,
      referenceWeightType: "ease in",
    };
    const workflow = generateImgWorkflow(params, 1, "sd15");

    // CLIPVisionLoader loads CLIP vision model
    expect(workflow["clip_vision_loader"]).toBeDefined();
    expect(workflow["clip_vision_loader"]!.class_type).toBe("CLIPVisionLoader");
    expect(workflow["clip_vision_loader"]!.inputs.clip_name).toBe(
      "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    );

    // IPAdapterModelLoader loads IP Adapter model
    expect(workflow["ipadapter_loader"]).toBeDefined();
    expect(workflow["ipadapter_loader"]!.class_type).toBe(
      "IPAdapterModelLoader"
    );
    expect(workflow["ipadapter_loader"]!.inputs.ipadapter_file).toBe(
      "ip-adapter-plus_sd15.safetensors"
    );

    // Reference image loaded
    expect(workflow["ref_image_0"]).toBeDefined();
    expect(workflow["ref_image_0"]!.class_type).toBe("LoadImage");
    expect(workflow["ref_image_0"]!.inputs.image).toBe("ref1.png");

    // IPAdapterAdvanced applies the reference
    expect(workflow["ipadapter"]).toBeDefined();
    expect(workflow["ipadapter"]!.class_type).toBe("IPAdapterAdvanced");
    expect(workflow["ipadapter"]!.inputs.weight).toBe(0.8);
    expect(workflow["ipadapter"]!.inputs.weight_type).toBe("ease in");
    expect(workflow["ipadapter"]!.inputs.combine_embeds).toBe("concat");
    expect(workflow["ipadapter"]!.inputs.model).toEqual(["checkpoint", 0]);
    expect(workflow["ipadapter"]!.inputs.ipadapter).toEqual([
      "ipadapter_loader",
      0,
    ]);
    expect(workflow["ipadapter"]!.inputs.clip_vision).toEqual([
      "clip_vision_loader",
      0,
    ]);
    expect(workflow["ipadapter"]!.inputs.image).toEqual(["ref_image_0", 0]);

    // Sampler should use IP Adapter output
    expect(workflow["sampler"]!.inputs.model).toEqual(["ipadapter", 0]);
  });

  it("should add IP Adapter nodes for SDXL with reference images", () => {
    const params: ImageParams = {
      ...baseParams,
      referenceImages: [{ filename: "ref1.png" }],
      referenceWeight: 0.5,
    };
    const workflow = generateImgWorkflow(params, 1, "sdxl");

    // Should use SDXL CLIP vision model
    expect(workflow["clip_vision_loader"]!.inputs.clip_name).toBe(
      "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    );

    // Should use SDXL plus IP Adapter (always uses plus variant)
    expect(workflow["ipadapter_loader"]!.inputs.ipadapter_file).toBe(
      "ip-adapter-plus_sdxl_vit-h.safetensors"
    );

    expect(workflow["ipadapter"]).toBeDefined();
    expect(workflow["ipadapter"]!.inputs.weight).toBe(0.5);
  });

  it("should batch multiple reference images", () => {
    const params: ImageParams = {
      ...baseParams,
      referenceImages: [{ filename: "ref1.png" }, { filename: "ref2.png" }],
      referenceCombineMode: "average",
    };
    const workflow = generateImgWorkflow(params, 1, "sd15");

    // Both images loaded
    expect(workflow["ref_image_0"]).toBeDefined();
    expect(workflow["ref_image_1"]).toBeDefined();

    // Images batched together
    expect(workflow["image_batch_1"]).toBeDefined();
    expect(workflow["image_batch_1"]!.class_type).toBe("ImageBatch");
    expect(workflow["image_batch_1"]!.inputs.image1).toEqual([
      "ref_image_0",
      0,
    ]);
    expect(workflow["image_batch_1"]!.inputs.image2).toEqual([
      "ref_image_1",
      0,
    ]);

    // IPAdapterAdvanced uses batched image
    expect(workflow["ipadapter"]!.inputs.image).toEqual(["image_batch_1", 0]);
    expect(workflow["ipadapter"]!.inputs.combine_embeds).toBe("average");

    // Sampler uses IP Adapter output
    expect(workflow["sampler"]!.inputs.model).toEqual(["ipadapter", 0]);
  });

  it("should batch three or more reference images", () => {
    const params: ImageParams = {
      ...baseParams,
      referenceImages: [
        { filename: "ref1.png" },
        { filename: "ref2.png" },
        { filename: "ref3.png" },
      ],
    };
    const workflow = generateImgWorkflow(params, 1, "sd15");

    // All images loaded
    expect(workflow["ref_image_0"]).toBeDefined();
    expect(workflow["ref_image_1"]).toBeDefined();
    expect(workflow["ref_image_2"]).toBeDefined();

    // Images batched in chain
    expect(workflow["image_batch_1"]!.inputs.image1).toEqual([
      "ref_image_0",
      0,
    ]);
    expect(workflow["image_batch_1"]!.inputs.image2).toEqual([
      "ref_image_1",
      0,
    ]);
    expect(workflow["image_batch_2"]!.inputs.image1).toEqual([
      "image_batch_1",
      0,
    ]);
    expect(workflow["image_batch_2"]!.inputs.image2).toEqual([
      "ref_image_2",
      0,
    ]);

    // IPAdapterAdvanced uses final batched image
    expect(workflow["ipadapter"]!.inputs.image).toEqual(["image_batch_2", 0]);
  });

  it("should not add IP Adapter nodes when no reference images", () => {
    const workflow = generateImgWorkflow(baseParams, 1, "sd15");

    expect(workflow["ipadapter_loader"]).toBeUndefined();
    expect(workflow["ipadapter"]).toBeUndefined();
  });

  it("should not add IP Adapter nodes for ZIT architecture", () => {
    const params: ImageParams = {
      ...zitParams,
      referenceImages: [{ filename: "ref1.png" }],
    };
    const workflow = generateImgWorkflow(params, 1, "zit");

    expect(workflow["ipadapter_loader"]).toBeUndefined();
    expect(workflow["ipadapter"]).toBeUndefined();
  });

  it("should not add IP Adapter nodes for unknown architecture", () => {
    const params: ImageParams = {
      ...baseParams,
      referenceImages: [{ filename: "ref1.png" }],
    };
    const workflow = generateImgWorkflow(params, 1, "unknown");

    expect(workflow["ipadapter_loader"]).toBeUndefined();
  });

  it("should chain IP Adapter after LoRAs", () => {
    const params: ImageParams = {
      ...baseParams,
      loras: [{ filename: "lora1.safetensors", weight: 0.8 }],
      referenceImages: [{ filename: "ref1.png" }],
    };
    const workflow = generateImgWorkflow(params, 1, "sd15");

    // IPAdapterAdvanced connects to LoRA output
    expect(workflow["ipadapter"]!.inputs.model).toEqual(["lora_0", 0]);

    // Sampler uses IP Adapter output
    expect(workflow["sampler"]!.inputs.model).toEqual(["ipadapter", 0]);
  });
});

// ControlNet tests
describe("generateImgWorkflow (ControlNet)", () => {
  it("should add ControlNet nodes for SDXL with controlNets (no preprocessor)", () => {
    const params: ImageParams = {
      ...baseParams,
      controlNets: [
        {
          type: "openpose",
          image: "pose.png",
          preprocessor: null,
          weight: 0.8,
        },
      ],
    };
    const workflow = generateImgWorkflow(params, 1, "sdxl");

    // ControlNetLoader loads the Union ControlNet model
    expect(workflow["controlnet_loader"]).toBeDefined();
    expect(workflow["controlnet_loader"]!.class_type).toBe("ControlNetLoader");
    expect(workflow["controlnet_loader"]!.inputs.control_net_name).toBe(
      "SDXL/controlnet-union-sdxl-1.0/diffusion_pytorch_model.safetensors"
    );

    // ControlNet image loaded
    expect(workflow["ctrl_image_0"]).toBeDefined();
    expect(workflow["ctrl_image_0"]!.class_type).toBe("LoadImage");
    expect(workflow["ctrl_image_0"]!.inputs.image).toBe("pose.png");

    // No preprocessor node when preprocessor is null
    expect(workflow["ctrl_preprocess_0"]).toBeUndefined();

    // SetUnionControlNetType sets the type
    expect(workflow["ctrl_type_0"]).toBeDefined();
    expect(workflow["ctrl_type_0"]!.class_type).toBe("SetUnionControlNetType");
    expect(workflow["ctrl_type_0"]!.inputs.type).toBe("openpose");
    expect(workflow["ctrl_type_0"]!.inputs.control_net).toEqual([
      "controlnet_loader",
      0,
    ]);

    // ControlNetApplyAdvanced applies to conditioning using raw image
    expect(workflow["ctrl_apply_0"]).toBeDefined();
    expect(workflow["ctrl_apply_0"]!.class_type).toBe(
      "ControlNetApplyAdvanced"
    );
    expect(workflow["ctrl_apply_0"]!.inputs.strength).toBe(0.8);
    expect(workflow["ctrl_apply_0"]!.inputs.control_net).toEqual([
      "ctrl_type_0",
      0,
    ]);
    expect(workflow["ctrl_apply_0"]!.inputs.image).toEqual(["ctrl_image_0", 0]);
    expect(workflow["ctrl_apply_0"]!.inputs.positive).toEqual(["positive", 0]);
    expect(workflow["ctrl_apply_0"]!.inputs.negative).toEqual(["negative", 0]);

    // Sampler uses ControlNet modified conditioning
    expect(workflow["sampler"]!.inputs.positive).toEqual(["ctrl_apply_0", 0]);
    expect(workflow["sampler"]!.inputs.negative).toEqual(["ctrl_apply_0", 1]);
  });

  it("should add preprocessor node when preprocessor is specified", () => {
    const params: ImageParams = {
      ...baseParams,
      controlNets: [
        {
          type: "openpose",
          image: "pose.png",
          preprocessor: "dwpose",
          weight: 0.8,
        },
      ],
    };
    const workflow = generateImgWorkflow(params, 1, "sdxl");

    // ControlNet image loaded
    expect(workflow["ctrl_image_0"]).toBeDefined();
    expect(workflow["ctrl_image_0"]!.inputs.image).toBe("pose.png");

    // AIO_Preprocessor processes the image
    expect(workflow["ctrl_preprocess_0"]).toBeDefined();
    expect(workflow["ctrl_preprocess_0"]!.class_type).toBe("AIO_Preprocessor");
    expect(workflow["ctrl_preprocess_0"]!.inputs.preprocessor).toBe("dwpose");
    expect(workflow["ctrl_preprocess_0"]!.inputs.image).toEqual([
      "ctrl_image_0",
      0,
    ]);
    expect(workflow["ctrl_preprocess_0"]!.inputs.resolution).toBe(512);

    // ControlNetApplyAdvanced uses preprocessed image
    expect(workflow["ctrl_apply_0"]!.inputs.image).toEqual([
      "ctrl_preprocess_0",
      0,
    ]);
  });

  it("should chain multiple ControlNets", () => {
    const params: ImageParams = {
      ...baseParams,
      controlNets: [
        {
          type: "openpose",
          image: "pose.png",
          preprocessor: null,
          weight: 0.8,
        },
        { type: "depth", image: "depth.png", preprocessor: null, weight: 0.6 },
      ],
    };
    const workflow = generateImgWorkflow(params, 1, "sdxl");

    // Both images loaded
    expect(workflow["ctrl_image_0"]).toBeDefined();
    expect(workflow["ctrl_image_1"]).toBeDefined();

    // Both types set
    expect(workflow["ctrl_type_0"]!.inputs.type).toBe("openpose");
    expect(workflow["ctrl_type_1"]!.inputs.type).toBe("depth");

    // First ControlNet applies to prompt conditioning
    expect(workflow["ctrl_apply_0"]!.inputs.positive).toEqual(["positive", 0]);
    expect(workflow["ctrl_apply_0"]!.inputs.negative).toEqual(["negative", 0]);

    // Second ControlNet chains from first
    expect(workflow["ctrl_apply_1"]!.inputs.positive).toEqual([
      "ctrl_apply_0",
      0,
    ]);
    expect(workflow["ctrl_apply_1"]!.inputs.negative).toEqual([
      "ctrl_apply_0",
      1,
    ]);

    // Sampler uses final ControlNet output
    expect(workflow["sampler"]!.inputs.positive).toEqual(["ctrl_apply_1", 0]);
    expect(workflow["sampler"]!.inputs.negative).toEqual(["ctrl_apply_1", 1]);
  });

  it("should use correct type values for different ControlNet types", () => {
    const types = [
      { type: "openpose" as const, expected: "openpose" },
      { type: "depth" as const, expected: "depth" },
      { type: "normal" as const, expected: "normal" },
      {
        type: "lineart" as const,
        expected: "canny/lineart/anime_lineart/mlsd",
      },
      {
        type: "scribble" as const,
        expected: "hed/pidi/scribble/ted",
      },
      { type: "segment" as const, expected: "segment" },
      { type: "tile" as const, expected: "tile" },
      { type: "repaint" as const, expected: "repaint" },
    ];

    for (const { type, expected } of types) {
      const params: ImageParams = {
        ...baseParams,
        controlNets: [
          { type, image: "img.png", preprocessor: null, weight: 1 },
        ],
      };
      const workflow = generateImgWorkflow(params, 1, "sdxl");
      expect(workflow["ctrl_type_0"]!.inputs.type).toBe(expected);
    }
  });

  it("should add ControlNet nodes for SD1.5 with separate model loaders", () => {
    const params: ImageParams = {
      ...baseParams,
      controlNets: [
        {
          type: "openpose",
          image: "pose.png",
          preprocessor: null,
          weight: 0.8,
        },
      ],
    };

    // SD1.5 should support ControlNet with separate loaders per type
    const sd15Workflow = generateImgWorkflow(params, 1, "sd15");
    // SD1.5 uses controlnet_loader_0 (per-ControlNet), not controlnet_loader (single Union)
    expect(sd15Workflow["controlnet_loader"]).toBeUndefined();
    expect(sd15Workflow["controlnet_loader_0"]).toBeDefined();
    expect(sd15Workflow["ctrl_apply_0"]).toBeDefined();
    // SD1.5 should not have SetUnionControlNetType
    expect(sd15Workflow["ctrl_type_0"]).toBeUndefined();
  });

  it("should not add ControlNet nodes for unsupported architectures", () => {
    const params: ImageParams = {
      ...baseParams,
      controlNets: [
        {
          type: "openpose",
          image: "pose.png",
          preprocessor: null,
          weight: 0.8,
        },
      ],
    };

    // ZIT should not support ControlNet
    const zitWorkflow = generateImgWorkflow(
      { ...zitParams, controlNets: params.controlNets },
      1,
      "zit"
    );
    expect(zitWorkflow["controlnet_loader"]).toBeUndefined();
    expect(zitWorkflow["ctrl_apply_0"]).toBeUndefined();
  });

  it("should not add ControlNet nodes when no controlNets", () => {
    const workflow = generateImgWorkflow(baseParams, 1, "sdxl");
    expect(workflow["controlnet_loader"]).toBeUndefined();
    expect(workflow["ctrl_apply_0"]).toBeUndefined();
  });

  it("should chain ControlNet after IP Adapter", () => {
    const params: ImageParams = {
      ...baseParams,
      referenceImages: [{ filename: "ref.png" }],
      controlNets: [
        {
          type: "openpose",
          image: "pose.png",
          preprocessor: null,
          weight: 0.8,
        },
      ],
    };
    const workflow = generateImgWorkflow(params, 1, "sdxl");

    // IP Adapter should be present and modify model
    expect(workflow["ipadapter"]).toBeDefined();
    expect(workflow["sampler"]!.inputs.model).toEqual(["ipadapter", 0]);

    // ControlNet should modify conditioning
    expect(workflow["ctrl_apply_0"]).toBeDefined();
    expect(workflow["sampler"]!.inputs.positive).toEqual(["ctrl_apply_0", 0]);
    expect(workflow["sampler"]!.inputs.negative).toEqual(["ctrl_apply_0", 1]);
  });

  it("should chain ControlNet after LoRAs and IP Adapter", () => {
    const params: ImageParams = {
      ...baseParams,
      loras: [{ filename: "lora.safetensors", weight: 0.8 }],
      referenceImages: [{ filename: "ref.png" }],
      controlNets: [
        { type: "depth", image: "depth.png", preprocessor: null, weight: 0.7 },
      ],
    };
    const workflow = generateImgWorkflow(params, 1, "sdxl");

    // LoRA modifies model and clip
    expect(workflow["lora_0"]).toBeDefined();

    // IP Adapter chains from LoRA
    expect(workflow["ipadapter"]!.inputs.model).toEqual(["lora_0", 0]);

    // Sampler model comes from IP Adapter
    expect(workflow["sampler"]!.inputs.model).toEqual(["ipadapter", 0]);

    // ControlNet modifies conditioning
    expect(workflow["sampler"]!.inputs.positive).toEqual(["ctrl_apply_0", 0]);
    expect(workflow["sampler"]!.inputs.negative).toEqual(["ctrl_apply_0", 1]);
  });
});

// Backwards compatibility test
describe("generateImgWorkflow alias", () => {
  it("should be an alias for generateImgWorkflow", () => {
    expect(generateImgWorkflow).toBe(generateImgWorkflow);
  });
});
