import { Box, Flex, IconButton, Text, Tooltip } from "@radix-ui/themes";
import type {
  ImageMetadata,
  ImageParams,
  LoRAConfig,
  SingleOperationRecord,
} from "../../shared/types/image";
import type { ControlNetConfig } from "../../shared/types/controlnet";
import type { ModelInfo } from "../../shared/types/models";
import { CONTROLNET_TYPE_LABELS } from "../../shared/constants/controlnet";
import { formatPreprocessorName } from "../../shared/utils";
import { ImageThumbnail } from "./ImageThumbnail";

interface ImageMetadataDisplayProps {
  metadata: ImageMetadata;
  checkpoints: ModelInfo[];
  loras: ModelInfo[];
  currentLoras: LoRAConfig[];
  currentControlNets: ControlNetConfig[];
  onUseParams: (params: Partial<ImageParams>, label: string) => void;
  onUseAspectRatio: () => void;
  onUseSourceImage: (filename: string) => void;
  onUseReferenceImage: (filename: string) => void;
  onUseControlNetImage: (filename: string, index: number) => void;
  onInsertPromptFragment?: (
    text: string,
    target: "prompt" | "negative"
  ) => void;
}

// Get display label for ControlNet config
function getControlNetLabel(config: ControlNetConfig): string {
  if (config.preprocessor) {
    return formatPreprocessorName(config.preprocessor);
  }
  return `${CONTROLNET_TYPE_LABELS[config.type] || config.type} (Preprocessed)`;
}

// Generate metadata display component (reused in sequences)
interface GenerateMetadataDisplayProps {
  params: ImageParams;
  checkpoints: ModelInfo[];
  loras: ModelInfo[];
  currentLoras: LoRAConfig[];
  currentControlNets: ControlNetConfig[];
  onUseParams: (params: Partial<ImageParams>, label: string) => void;
  onUseAspectRatio: () => void;
  onUseSourceImage: (filename: string) => void;
  onUseReferenceImage: (filename: string) => void;
  onUseControlNetImage: (filename: string, index: number) => void;
  onInsertPromptFragment?: (
    text: string,
    target: "prompt" | "negative"
  ) => void;
}

function GenerateMetadataDisplay({
  params,
  checkpoints,
  loras,
  currentLoras,
  currentControlNets,
  onUseParams,
  onUseAspectRatio,
  onUseSourceImage,
  onUseReferenceImage,
  onUseControlNetImage,
  onInsertPromptFragment,
}: GenerateMetadataDisplayProps) {
  const getCheckpointTitle = (filename: string): string => {
    return checkpoints.find((c) => c.filename === filename)?.title || filename;
  };

  const getLoraTitle = (filename: string): string => {
    return loras.find((l) => l.filename === filename)?.title || filename;
  };

  const handleUseSingleLora = (lora: LoRAConfig) => {
    const existingIndex = currentLoras.findIndex(
      (l) => l.filename === lora.filename
    );
    let newLoras: LoRAConfig[];
    if (existingIndex >= 0) {
      newLoras = currentLoras.map((l, i) => (i === existingIndex ? lora : l));
    } else {
      newLoras = [...currentLoras, lora];
    }
    onUseParams({ loras: newLoras }, `LoRA: ${getLoraTitle(lora.filename)}`);
  };

  return (
    <>
      {/* Prompt row */}
      <Flex gap="4" className="border-b border-[var(--gray-a4)] pb-2">
        <Tooltip content="Use entire prompt">
          <IconButton
            size="1"
            variant="ghost"
            color="gray"
            className="w-16 shrink-0 justify-start self-center"
            onClick={() => onUseParams({ prompt: params.prompt }, "prompt")}
          >
            <Text size="1" className="text-[var(--gray-10)]">
              Prompt
            </Text>
          </IconButton>
        </Tooltip>
        <Box className="flex-1 rounded p-1">
          <PromptFragments
            text={params.prompt}
            onClickFragment={
              onInsertPromptFragment
                ? (fragment) => onInsertPromptFragment(fragment, "prompt")
                : undefined
            }
          />
        </Box>
      </Flex>

      {/* Negative Prompt row */}
      {params.negativePrompt && (
        <Flex gap="4" mt="1" className="border-b border-[var(--gray-a4)] pb-2">
          <Tooltip content="Use entire negative prompt">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              className="w-16 shrink-0 justify-start self-center"
              onClick={() =>
                onUseParams(
                  { negativePrompt: params.negativePrompt },
                  "negative prompt"
                )
              }
            >
              <Text size="1" className="text-[var(--gray-10)]">
                Negative
              </Text>
            </IconButton>
          </Tooltip>
          <Box className="flex-1 rounded p-1">
            <PromptFragments
              text={params.negativePrompt}
              isNegative
              onClickFragment={
                onInsertPromptFragment
                  ? (fragment) => onInsertPromptFragment(fragment, "negative")
                  : undefined
              }
            />
          </Box>
        </Flex>
      )}

      {/* Row 1: General (size + seed) */}
      <Flex gap="4" mt="2" className="border-b border-[var(--gray-a4)] pb-1">
        <Tooltip content="Use size and seed">
          <IconButton
            size="1"
            variant="ghost"
            color="gray"
            className="w-16 shrink-0 justify-start self-center"
            onClick={() =>
              onUseParams(
                {
                  width: params.width,
                  height: params.height,
                  seed: params.seed,
                },
                "general settings"
              )
            }
          >
            <Text size="1" className="text-[var(--gray-10)]">
              General
            </Text>
          </IconButton>
        </Tooltip>
        <Flex gap="2" align="center" wrap="wrap" className="flex-1">
          <Tooltip content="Use aspect ratio">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              onClick={onUseAspectRatio}
            >
              <Text size="1">
                <Text className="text-[var(--gray-10)]">Size: </Text>
                {params.width}×{params.height}
              </Text>
            </IconButton>
          </Tooltip>
          <Tooltip content="Use seed">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              onClick={() => onUseParams({ seed: params.seed }, "seed")}
            >
              <Text size="1">
                <Text className="text-[var(--gray-10)]">Seed: </Text>
                {params.seed}
              </Text>
            </IconButton>
          </Tooltip>
        </Flex>
      </Flex>

      {/* Row 2: Sampling */}
      <Flex gap="4" mt="1" className="border-b border-[var(--gray-a4)] pb-1">
        <Tooltip content="Use all sampling settings">
          <IconButton
            size="1"
            variant="ghost"
            color="gray"
            className="w-16 shrink-0 justify-start self-center"
            onClick={() =>
              onUseParams(
                {
                  steps: params.steps,
                  cfgScale: params.cfgScale,
                  sampler: params.sampler,
                  scheduler: params.scheduler,
                },
                "sampling settings"
              )
            }
          >
            <Text size="1" className="text-[var(--gray-10)]">
              Sampling
            </Text>
          </IconButton>
        </Tooltip>
        <Flex gap="2" align="center" wrap="wrap" className="flex-1">
          <Tooltip content="Use steps">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              onClick={() => onUseParams({ steps: params.steps }, "steps")}
            >
              <Text size="1">
                <Text className="text-[var(--gray-10)]">Steps: </Text>
                {params.steps}
              </Text>
            </IconButton>
          </Tooltip>
          <Tooltip content="Use CFG">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              onClick={() => onUseParams({ cfgScale: params.cfgScale }, "CFG")}
            >
              <Text size="1">
                <Text className="text-[var(--gray-10)]">CFG: </Text>
                {params.cfgScale}
              </Text>
            </IconButton>
          </Tooltip>
          <Tooltip content="Use sampler and scheduler">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              onClick={() =>
                onUseParams(
                  { sampler: params.sampler, scheduler: params.scheduler },
                  "sampler"
                )
              }
            >
              <Text size="1">
                <Text className="text-[var(--gray-10)]">Sampler: </Text>
                {params.sampler}/{params.scheduler}
              </Text>
            </IconButton>
          </Tooltip>
        </Flex>
      </Flex>

      {/* Row 3: Model */}
      {params.model && (
        <Flex gap="4" mt="1" className="border-b border-[var(--gray-a4)] pb-1">
          <Tooltip content="Use checkpoint">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              className="w-16 shrink-0 justify-start self-center"
              onClick={() => onUseParams({ model: params.model }, "checkpoint")}
            >
              <Text size="1" className="text-[var(--gray-10)]">
                Model
              </Text>
            </IconButton>
          </Tooltip>
          <Flex align="center" className="flex-1">
            <Tooltip content="Use checkpoint">
              <IconButton
                size="1"
                variant="ghost"
                color="gray"
                onClick={() =>
                  onUseParams({ model: params.model }, "checkpoint")
                }
              >
                <Text size="1">{getCheckpointTitle(params.model)}</Text>
              </IconButton>
            </Tooltip>
          </Flex>
        </Flex>
      )}

      {/* Row 4: LoRA */}
      {params.loras && params.loras.length > 0 && (
        <Flex gap="4" mt="1" className="border-b border-[var(--gray-a4)] pb-1">
          <Tooltip content="Use all LoRAs">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              className="w-16 shrink-0 justify-start self-center"
              onClick={() => onUseParams({ loras: params.loras }, "LoRAs")}
            >
              <Text size="1" className="text-[var(--gray-10)]">
                LoRA
              </Text>
            </IconButton>
          </Tooltip>
          <Flex gap="2" align="center" wrap="wrap" className="flex-1">
            {params.loras.map((lora, index) => (
              <Tooltip
                key={`${lora.filename}-${index}`}
                content={`Use ${getLoraTitle(lora.filename)}`}
              >
                <IconButton
                  size="1"
                  variant="ghost"
                  color="gray"
                  onClick={() => handleUseSingleLora(lora)}
                >
                  <Text size="1">
                    {getLoraTitle(lora.filename)}({lora.weight.toFixed(2)})
                  </Text>
                </IconButton>
              </Tooltip>
            ))}
          </Flex>
        </Flex>
      )}

      {/* Row 5: Source Image */}
      {params.sourceImage && (
        <Flex gap="4" mt="1" className="border-b border-[var(--gray-a4)] pb-1">
          <Tooltip content="Use source image settings">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              className="w-16 shrink-0 justify-start self-center"
              onClick={() =>
                onUseParams(
                  {
                    sourceImage: params.sourceImage,
                    sourceImageStrength: params.sourceImageStrength,
                  },
                  "source image settings"
                )
              }
            >
              <Text size="1" className="text-[var(--gray-10)]">
                Source
              </Text>
            </IconButton>
          </Tooltip>
          <Flex gap="2" align="center" wrap="wrap" className="min-h-10 flex-1">
            <Tooltip content="Use as source image">
              <Box>
                <ImageThumbnail
                  filename={params.sourceImage}
                  size="small"
                  selectable={true}
                  onClick={() => onUseSourceImage(params.sourceImage!)}
                />
              </Box>
            </Tooltip>
            <Tooltip content="Use source image strength">
              <IconButton
                size="1"
                variant="ghost"
                color="gray"
                onClick={() =>
                  onUseParams(
                    { sourceImageStrength: params.sourceImageStrength },
                    "source image strength"
                  )
                }
              >
                <Text size="1">
                  <Text className="text-[var(--gray-10)]">Strength: </Text>
                  {params.sourceImageStrength.toFixed(2)}
                </Text>
              </IconButton>
            </Tooltip>
          </Flex>
        </Flex>
      )}

      {/* Row 6: Reference Images */}
      {params.referenceImages && params.referenceImages.length > 0 && (
        <Flex gap="4" mt="1" className="border-b border-[var(--gray-a4)] pb-1">
          <Tooltip content="Use reference settings">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              className="w-16 shrink-0 justify-start self-center"
              onClick={() =>
                onUseParams(
                  {
                    referenceImages: params.referenceImages,
                    referenceWeight: params.referenceWeight,
                    referenceWeightType: params.referenceWeightType,
                    referenceCombineMode: params.referenceCombineMode,
                  },
                  "reference settings"
                )
              }
            >
              <Text size="1" className="text-[var(--gray-10)]">
                Reference
              </Text>
            </IconButton>
          </Tooltip>
          <Flex gap="2" align="center" wrap="wrap" className="min-h-10 flex-1">
            {params.referenceImages.map((ref, index) => (
              <Tooltip
                key={`${ref.filename}-${index}`}
                content="Use as reference image"
              >
                <Box>
                  <ImageThumbnail
                    filename={ref.filename}
                    size="small"
                    selectable={true}
                    onClick={() => onUseReferenceImage(ref.filename)}
                  />
                </Box>
              </Tooltip>
            ))}
            <Tooltip content="Use reference weight, type, and combine mode">
              <IconButton
                size="1"
                variant="ghost"
                color="gray"
                onClick={() =>
                  onUseParams(
                    {
                      referenceWeight: params.referenceWeight,
                      referenceWeightType: params.referenceWeightType,
                      referenceCombineMode: params.referenceCombineMode,
                    },
                    "reference settings"
                  )
                }
              >
                <Text size="1">
                  <Text className="text-[var(--gray-10)]">Weight: </Text>
                  {params.referenceWeight.toFixed(2)}{" "}
                  <Text className="text-[var(--gray-10)]">Type: </Text>
                  {params.referenceWeightType}
                  {params.referenceImages.length > 1 && (
                    <>
                      {" "}
                      <Text className="text-[var(--gray-10)]">Combine: </Text>
                      {params.referenceCombineMode}
                    </>
                  )}
                </Text>
              </IconButton>
            </Tooltip>
          </Flex>
        </Flex>
      )}

      {/* Row 7: ControlNets */}
      {params.controlNets && params.controlNets.length > 0 && (
        <Flex gap="4" mt="1" className="border-b border-[var(--gray-a4)] pb-1">
          <Tooltip content="Use all ControlNet settings">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              className="w-16 shrink-0 justify-start self-center"
              onClick={() =>
                onUseParams({ controlNets: params.controlNets }, "ControlNets")
              }
            >
              <Text size="1" className="text-[var(--gray-10)]">
                ControlNet
              </Text>
            </IconButton>
          </Tooltip>
          <Flex gap="2" align="center" wrap="wrap" className="min-h-10 flex-1">
            {params.controlNets.map((config, index) => (
              <Flex
                key={`${config.image}-${index}`}
                align="center"
                gap="1"
                className="rounded bg-[var(--gray-a3)] p-1"
              >
                <Tooltip content={`Use as ControlNet ${index + 1} image`}>
                  <Box>
                    <ImageThumbnail
                      filename={config.image}
                      size="small"
                      selectable={true}
                      onClick={() => onUseControlNetImage(config.image, index)}
                    />
                  </Box>
                </Tooltip>
                <Tooltip content={`Use ControlNet ${index + 1} settings`}>
                  <IconButton
                    size="1"
                    variant="ghost"
                    color="gray"
                    onClick={() =>
                      onUseParams(
                        {
                          controlNets: [
                            ...currentControlNets.slice(0, index),
                            config,
                            ...currentControlNets.slice(index + 1),
                          ],
                        },
                        `ControlNet ${index + 1}`
                      )
                    }
                  >
                    <Text size="1">
                      {getControlNetLabel(config)} ({config.weight.toFixed(2)})
                    </Text>
                  </IconButton>
                </Tooltip>
              </Flex>
            ))}
          </Flex>
        </Flex>
      )}

      {/* Postprocessing */}
      {(params.faceDetailer || params.upscaleEnabled) && (
        <Flex gap="4" mt="1" align="center">
          <Tooltip content="Use postprocessing settings">
            <IconButton
              size="1"
              variant="ghost"
              color="gray"
              className="w-16 shrink-0 justify-start self-center"
              onClick={() =>
                onUseParams(
                  {
                    faceDetailer: params.faceDetailer,
                    upscaleEnabled: params.upscaleEnabled,
                    upscaleFactor: params.upscaleFactor,
                  },
                  "postprocessing settings"
                )
              }
            >
              <Text size="1" className="text-[var(--gray-10)]">
                Postprocess
              </Text>
            </IconButton>
          </Tooltip>
          <Flex gap="2" align="center" wrap="wrap" className="flex-1">
            {params.faceDetailer && (
              <Tooltip content="Use Face Detailer">
                <IconButton
                  size="1"
                  variant="ghost"
                  color="gray"
                  onClick={() =>
                    onUseParams(
                      { faceDetailer: params.faceDetailer },
                      "Face Detailer"
                    )
                  }
                >
                  <Text size="1">Face Detailer</Text>
                </IconButton>
              </Tooltip>
            )}
            {params.upscaleEnabled && (
              <Tooltip content="Use upscale settings">
                <IconButton
                  size="1"
                  variant="ghost"
                  color="gray"
                  onClick={() =>
                    onUseParams(
                      {
                        upscaleEnabled: params.upscaleEnabled,
                        upscaleFactor: params.upscaleFactor,
                      },
                      "upscale settings"
                    )
                  }
                >
                  <Text size="1">
                    <Text className="text-[var(--gray-10)]">Upscale: </Text>
                    {params.upscaleFactor}×
                  </Text>
                </IconButton>
              </Tooltip>
            )}
          </Flex>
        </Flex>
      )}
    </>
  );
}

// Inline operation details for a single operation (used in sequences)
interface OperationDetailsProps {
  operation: SingleOperationRecord;
  checkpoints: ModelInfo[];
  loras: ModelInfo[];
  currentLoras: LoRAConfig[];
  currentControlNets: ControlNetConfig[];
  onUseParams: (params: Partial<ImageParams>, label: string) => void;
  onUseAspectRatio: () => void;
  onUseSourceImage: (filename: string) => void;
  onUseReferenceImage: (filename: string) => void;
  onUseControlNetImage: (filename: string, index: number) => void;
  onInsertPromptFragment?: (
    text: string,
    target: "prompt" | "negative"
  ) => void;
}

function OperationDetails({
  operation,
  checkpoints,
  loras,
  currentLoras,
  currentControlNets,
  onUseParams,
  onUseAspectRatio,
  onUseSourceImage,
  onUseReferenceImage,
  onUseControlNetImage,
  onInsertPromptFragment,
}: OperationDetailsProps) {
  const getCheckpointTitle = (filename: string): string => {
    return checkpoints.find((c) => c.filename === filename)?.title || filename;
  };

  const getLoraTitle = (filename: string): string => {
    return loras.find((l) => l.filename === filename)?.title || filename;
  };

  switch (operation.type) {
    case "raw":
      return (
        <Flex gap="4" align="center">
          <Text size="1" className="w-16 shrink-0 text-[var(--gray-10)]">
            Raw
          </Text>
          <Text size="1">Imported image</Text>
        </Flex>
      );

    case "generate":
      return (
        <GenerateMetadataDisplay
          params={operation.params}
          checkpoints={checkpoints}
          loras={loras}
          currentLoras={currentLoras}
          currentControlNets={currentControlNets}
          onUseParams={onUseParams}
          onUseAspectRatio={onUseAspectRatio}
          onUseSourceImage={onUseSourceImage}
          onUseReferenceImage={onUseReferenceImage}
          onUseControlNetImage={onUseControlNetImage}
          onInsertPromptFragment={onInsertPromptFragment}
        />
      );

    case "detect":
      return (
        <Flex gap="4" align="center">
          <Text size="1" className="w-16 shrink-0 text-[var(--gray-10)]">
            Detect
          </Text>
          <Flex gap="2" align="center" wrap="wrap" className="flex-1">
            <Tooltip content="Use as source image">
              <Box>
                <ImageThumbnail
                  filename={operation.params.source}
                  size="small"
                  selectable={true}
                  onClick={() => onUseSourceImage(operation.params.source)}
                />
              </Box>
            </Tooltip>
            <Text size="1">
              <Text className="text-[var(--gray-10)]">Preprocessor: </Text>
              {formatPreprocessorName(operation.params.detector)}
            </Text>
            <Text size="1">
              <Text className="text-[var(--gray-10)]">Resolution: </Text>
              {operation.params.resolution}
            </Text>
          </Flex>
        </Flex>
      );

    case "flatten":
      return (
        <Flex gap="4" align="center">
          <Text size="1" className="w-16 shrink-0 text-[var(--gray-10)]">
            Flatten
          </Text>
          <Text size="1">{operation.layers.length} layers merged</Text>
        </Flex>
      );

    case "enhance": {
      const params = operation.params;
      const enhanceTypeLabel =
        params.enhanceType === "face_detailer" ? "Face Detailer" : "Upscale";

      if (params.enhanceType === "upscale") {
        return (
          <Flex gap="4" align="center">
            <Text size="1" className="w-16 shrink-0 text-[var(--gray-10)]">
              {enhanceTypeLabel}
            </Text>
            <Flex gap="2" align="center" wrap="wrap" className="flex-1">
              <Tooltip content="Use as source image">
                <Box>
                  <ImageThumbnail
                    filename={params.source}
                    size="small"
                    selectable={true}
                    onClick={() => onUseSourceImage(params.source)}
                  />
                </Box>
              </Tooltip>
              {params.upscaleFactor && (
                <Text size="1">
                  <Text className="text-[var(--gray-10)]">Factor: </Text>
                  {params.upscaleFactor}×
                </Text>
              )}
            </Flex>
          </Flex>
        );
      }

      // Face detailer - show more details
      return (
        <Flex direction="column" gap="1">
          <Flex gap="4" align="center">
            <Text size="1" className="w-16 shrink-0 text-[var(--gray-10)]">
              {enhanceTypeLabel}
            </Text>
            <Flex gap="2" align="center" wrap="wrap" className="flex-1">
              <Tooltip content="Use as source image">
                <Box>
                  <ImageThumbnail
                    filename={params.source}
                    size="small"
                    selectable={true}
                    onClick={() => onUseSourceImage(params.source)}
                  />
                </Box>
              </Tooltip>
              <Text size="1">
                <Text className="text-[var(--gray-10)]">Steps: </Text>
                {params.steps}
              </Text>
              <Text size="1">
                <Text className="text-[var(--gray-10)]">CFG: </Text>
                {params.cfgScale}
              </Text>
              <Text size="1">
                <Text className="text-[var(--gray-10)]">Seed: </Text>
                {params.seed}
              </Text>
            </Flex>
          </Flex>
          <Flex gap="4">
            <Text size="1" className="w-16 shrink-0 text-[var(--gray-10)]">
              Model
            </Text>
            <Text size="1">{getCheckpointTitle(params.model)}</Text>
          </Flex>
          {params.loras && params.loras.length > 0 && (
            <Flex gap="4">
              <Text size="1" className="w-16 shrink-0 text-[var(--gray-10)]">
                LoRA
              </Text>
              <Flex gap="2" wrap="wrap" className="flex-1">
                {params.loras.map((lora, index) => (
                  <Text key={`${lora.filename}-${index}`} size="1">
                    {getLoraTitle(lora.filename)}({lora.weight.toFixed(2)})
                  </Text>
                ))}
              </Flex>
            </Flex>
          )}
        </Flex>
      );
    }
  }
}

// Sequence metadata display component
interface SequenceMetadataDisplayProps {
  operations: SingleOperationRecord[];
  checkpoints: ModelInfo[];
  loras: ModelInfo[];
  currentLoras: LoRAConfig[];
  currentControlNets: ControlNetConfig[];
  onUseParams: (params: Partial<ImageParams>, label: string) => void;
  onUseAspectRatio: () => void;
  onUseSourceImage: (filename: string) => void;
  onUseReferenceImage: (filename: string) => void;
  onUseControlNetImage: (filename: string, index: number) => void;
  onInsertPromptFragment?: (
    text: string,
    target: "prompt" | "negative"
  ) => void;
}

function SequenceMetadataDisplay({
  operations,
  checkpoints,
  loras,
  currentLoras,
  currentControlNets,
  onUseParams,
  onUseAspectRatio,
  onUseSourceImage,
  onUseReferenceImage,
  onUseControlNetImage,
  onInsertPromptFragment,
}: SequenceMetadataDisplayProps) {
  return (
    <Box
      className="absolute bottom-0 left-0 right-0 max-h-[50%] overflow-y-auto p-3"
      style={{ backgroundColor: "rgba(0, 0, 0, 0.75)" }}
    >
      {operations.map((op, index) => (
        <Box
          key={index}
          className={
            index < operations.length - 1
              ? "mb-2 border-b border-[var(--gray-a6)] pb-2"
              : ""
          }
        >
          <OperationDetails
            operation={op}
            checkpoints={checkpoints}
            loras={loras}
            currentLoras={currentLoras}
            currentControlNets={currentControlNets}
            onUseParams={onUseParams}
            onUseAspectRatio={onUseAspectRatio}
            onUseSourceImage={onUseSourceImage}
            onUseReferenceImage={onUseReferenceImage}
            onUseControlNetImage={onUseControlNetImage}
            onInsertPromptFragment={onInsertPromptFragment}
          />
        </Box>
      ))}
    </Box>
  );
}

// Render prompt with clickable fragments (newline-separated lines)
interface PromptFragmentsProps {
  text: string;
  isNegative?: boolean;
  onClickFragment?: (fragment: string) => void;
}

function PromptFragments({
  text,
  isNegative = false,
  onClickFragment,
}: PromptFragmentsProps) {
  const trimmedText = text.trim();
  const fragments = trimmedText.split("\n");

  return (
    <Text
      size={isNegative ? "1" : "2"}
      className={isNegative ? "text-red-400" : "text-white"}
    >
      {fragments.map((fragment, index) => (
        <span key={index}>
          {index > 0 && <Text className="text-[var(--gray-8)]"> ↵ </Text>}
          {onClickFragment ? (
            <Tooltip content="Insert line into prompt">
              <span
                className="cursor-pointer rounded px-0.5 hover:bg-[var(--gray-a4)]"
                onClick={(e) => {
                  e.stopPropagation();
                  onClickFragment(fragment);
                }}
              >
                {fragment}
              </span>
            </Tooltip>
          ) : (
            fragment
          )}
        </span>
      ))}
    </Text>
  );
}

export function ImageMetadataDisplay({
  metadata,
  checkpoints,
  loras,
  currentLoras,
  currentControlNets,
  onUseParams,
  onUseAspectRatio,
  onUseSourceImage,
  onUseReferenceImage,
  onUseControlNetImage,
  onInsertPromptFragment,
}: ImageMetadataDisplayProps) {
  // Sequence operations get their own container with scrolling
  if (metadata.operation.type === "sequence") {
    return (
      <SequenceMetadataDisplay
        operations={metadata.operation.operations}
        checkpoints={checkpoints}
        loras={loras}
        currentLoras={currentLoras}
        currentControlNets={currentControlNets}
        onUseParams={onUseParams}
        onUseAspectRatio={onUseAspectRatio}
        onUseSourceImage={onUseSourceImage}
        onUseReferenceImage={onUseReferenceImage}
        onUseControlNetImage={onUseControlNetImage}
        onInsertPromptFragment={onInsertPromptFragment}
      />
    );
  }

  // Single operations use OperationDetails
  return (
    <Box
      className="absolute bottom-0 left-0 right-0 p-3"
      style={{ backgroundColor: "rgba(0, 0, 0, 0.75)" }}
    >
      <OperationDetails
        operation={metadata.operation}
        checkpoints={checkpoints}
        loras={loras}
        currentLoras={currentLoras}
        currentControlNets={currentControlNets}
        onUseParams={onUseParams}
        onUseAspectRatio={onUseAspectRatio}
        onUseSourceImage={onUseSourceImage}
        onUseReferenceImage={onUseReferenceImage}
        onUseControlNetImage={onUseControlNetImage}
        onInsertPromptFragment={onInsertPromptFragment}
      />
    </Box>
  );
}
