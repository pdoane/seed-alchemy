import { describe, it, expect } from "vitest";
import { detectArchitecture, type SafetensorsHeader } from "./safetensors";

describe("safetensors", () => {
  describe("detectArchitecture", () => {
    describe("from modelspec.architecture metadata", () => {
      it("detects SDXL from modelspec", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { "modelspec.architecture": "stable-diffusion-xl-v1-base" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sdxl");
        expect(result.source).toBe("modelspec");
      });

      it("detects SDXL refiner from modelspec", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: {
            "modelspec.architecture": "stable-diffusion-xl-v1-refiner",
          },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sdxl-refiner");
        expect(result.source).toBe("modelspec");
      });

      it("detects SD1.5 from modelspec", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { "modelspec.architecture": "stable-diffusion-v1-5" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd15");
        expect(result.source).toBe("modelspec");
      });

      it("detects SD3 from modelspec", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { "modelspec.architecture": "sd-3-medium" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd3");
        expect(result.source).toBe("modelspec");
      });

      it("detects SD3.5 from modelspec", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { "modelspec.architecture": "sd-3.5-large" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd35");
        expect(result.source).toBe("modelspec");
      });

      it("detects Flux from modelspec", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { "modelspec.architecture": "flux-dev" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("flux");
        expect(result.source).toBe("modelspec");
      });
    });

    describe("from Kohya ss_base_model_version metadata", () => {
      it("detects SDXL from Kohya metadata", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { ss_base_model_version: "sd_xl_base_1.0" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sdxl");
        expect(result.source).toBe("kohya");
      });

      it("detects SD1.5 from Kohya metadata", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { ss_base_model_version: "sd_v1_5" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd15");
        expect(result.source).toBe("kohya");
      });

      it("detects SD2.0 from Kohya metadata", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { ss_base_model_version: "sd_v2" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd20");
        expect(result.source).toBe("kohya");
      });

      it("detects Flux from Kohya metadata", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { ss_base_model_version: "flux1-dev" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("flux");
        expect(result.source).toBe("kohya");
      });

      it("detects SD3 from Kohya metadata", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { ss_base_model_version: "sd3_medium" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd3");
        expect(result.source).toBe("kohya");
      });

      it("detects SD3.5 from Kohya metadata", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { ss_base_model_version: "sd3.5_large" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd35");
        expect(result.source).toBe("kohya");
      });

      it("detects ZIT from Kohya metadata", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { ss_base_model_version: "zimage" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("zit");
        expect(result.source).toBe("kohya");
      });

      it("detects SD2.1 from Kohya metadata with 768", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: { ss_base_model_version: "sd_v2_768" },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd21");
        expect(result.source).toBe("kohya");
      });
    });

    describe("from tensor inspection", () => {
      it("detects Flux from tensor names", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "double_blocks.0.img_attn.norm.key_norm.scale": {
              dtype: "F16",
              shape: [128],
              data_offsets: [0, 256],
            },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("flux");
        expect(result.source).toBe("tensor");
      });

      it("detects SD3 from tensor names", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "joint_blocks.0.context_block.attn.qkv.weight": {
              dtype: "F16",
              shape: [768, 768],
              data_offsets: [0, 1179648],
            },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd3");
        expect(result.source).toBe("tensor");
      });

      it("detects Stable Cascade from tensor names", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "clf.1.weight": {
              dtype: "F16",
              shape: [16, 1280],
              data_offsets: [0, 40960],
            },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("cascade");
        expect(result.source).toBe("tensor");
      });

      it("detects Flux LoRA from Kohya-style tensor names (lora_unet_ with double_blocks)", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": {
              dtype: "BF16",
              shape: [16, 3072],
              data_offsets: [0, 98304],
            },
            "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": {
              dtype: "BF16",
              shape: [3072, 16],
              data_offsets: [98304, 196608],
            },
            "lora_unet_single_blocks_0_linear1.lora_down.weight": {
              dtype: "BF16",
              shape: [16, 3072],
              data_offsets: [196608, 294912],
            },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("flux");
        expect(result.source).toBe("tensor");
      });

      it("detects ZIT LoRA from diffusers-style tensor names (diffusion_model.layers)", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "diffusion_model.layers.0.attention.to_q.lora_A.weight": {
              dtype: "BF16",
              shape: [16, 1536],
              data_offsets: [0, 49152],
            },
            "diffusion_model.layers.0.attention.to_q.lora_B.weight": {
              dtype: "BF16",
              shape: [1536, 16],
              data_offsets: [49152, 98304],
            },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("zit");
        expect(result.source).toBe("tensor");
      });

      it("detects SDXL LoRA from tensor names", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "lora_te1_text_model_encoder.weight": {
              dtype: "F16",
              shape: [768, 64],
              data_offsets: [0, 98304],
            },
            "lora_te2_text_model_encoder.weight": {
              dtype: "F16",
              shape: [1280, 64],
              data_offsets: [98304, 262144],
            },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sdxl");
        expect(result.source).toBe("tensor");
      });

      it("detects SD1.5 LoRA from tensor names", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "lora_te_text_model_encoder.weight": {
              dtype: "F16",
              shape: [768, 64],
              data_offsets: [0, 98304],
            },
            "lora_unet_down_blocks.weight": {
              dtype: "F16",
              shape: [320, 64],
              data_offsets: [98304, 139264],
            },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd15");
        expect(result.source).toBe("tensor");
      });

      it("detects SD1.5 from context dimension 768", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight":
              {
                dtype: "F16",
                shape: [320, 768],
                data_offsets: [0, 491520],
              },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd15");
        expect(result.source).toBe("tensor");
      });

      it("detects SD2.0 from context dimension 1024", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight":
              {
                dtype: "F16",
                shape: [320, 1024],
                data_offsets: [0, 655360],
              },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd20");
        expect(result.source).toBe("tensor");
      });

      it("detects SDXL from context dimension 2048", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight":
              {
                dtype: "F16",
                shape: [640, 2048],
                data_offsets: [0, 2621440],
              },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sdxl");
        expect(result.source).toBe("tensor");
      });

      it("detects Flux LoRA from diffusers-style tensor names (lora_A/lora_B)", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "transformer.double_blocks.0.img_attn.proj.lora_A.weight": {
              dtype: "BF16",
              shape: [16, 3072],
              data_offsets: [0, 98304],
            },
            "transformer.double_blocks.0.img_attn.proj.lora_B.weight": {
              dtype: "BF16",
              shape: [3072, 16],
              data_offsets: [98304, 196608],
            },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("flux");
        expect(result.source).toBe("tensor");
      });

      it("detects Flux LoRA from diffusers-style tensor names (lora_down/lora_up)", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "double_blocks.0.img_attn.proj.lora_down.weight": {
              dtype: "BF16",
              shape: [16, 3072],
              data_offsets: [0, 98304],
            },
            "double_blocks.0.img_attn.proj.lora_up.weight": {
              dtype: "BF16",
              shape: [3072, 16],
              data_offsets: [98304, 196608],
            },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("flux");
        expect(result.source).toBe("tensor");
      });

      it("detects SD3 LoRA from diffusers-style tensor names", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "transformer.joint_blocks.0.attn.to_q.lora_A.weight": {
              dtype: "F16",
              shape: [16, 768],
              data_offsets: [0, 24576],
            },
            "transformer.joint_blocks.0.attn.to_q.lora_B.weight": {
              dtype: "F16",
              shape: [768, 16],
              data_offsets: [24576, 49152],
            },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sd3");
        expect(result.source).toBe("tensor");
      });

      it("detects ZIT from context_refiner tensors", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "context_refiner.0.attention.to_q.weight": {
              dtype: "F16",
              shape: [768, 768],
              data_offsets: [0, 1179648],
            },
          },
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("zit");
        expect(result.source).toBe("tensor");
      });
    });

    describe("priority order", () => {
      it("prefers modelspec over Kohya metadata", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: {
            "modelspec.architecture": "stable-diffusion-xl-v1-base",
            ss_base_model_version: "sd_v1_5",
          },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sdxl");
        expect(result.source).toBe("modelspec");
      });

      it("prefers Kohya metadata over tensor inspection", () => {
        const header: SafetensorsHeader = {
          tensors: {
            "lora_te_text_model_encoder.weight": {
              dtype: "F16",
              shape: [768, 64],
              data_offsets: [0, 98304],
            },
          },
          metadata: {
            ss_base_model_version: "sd_xl_base_1.0",
          },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("sdxl");
        expect(result.source).toBe("kohya");
      });
    });

    describe("unknown architecture", () => {
      it("returns unknown for empty header", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: {},
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("unknown");
        expect(result.source).toBe("unknown");
      });

      it("returns unknown for unrecognized metadata", () => {
        const header: SafetensorsHeader = {
          tensors: {},
          metadata: {
            "modelspec.architecture": "some-future-model",
          },
        };
        const result = detectArchitecture(header);
        expect(result.architecture).toBe("unknown");
        expect(result.source).toBe("unknown");
      });
    });
  });
});
