import { useMemo, useRef } from "react";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  horizontalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  Box,
  Button,
  Callout,
  DropdownMenu,
  Flex,
  Heading,
  IconButton,
  ScrollArea,
  Select,
  Slider,
  Switch,
  Text,
  TextArea,
  TextField,
} from "@radix-ui/themes";
import { PlusIcon, StopCircleIcon, XIcon } from "@phosphor-icons/react";
import { useImageStore } from "../store/imageStore";
import { usePromptKeyboard } from "../hooks/usePromptKeyboard";
import { ControlNetSettings } from "./ControlNetSettings";
import { DimensionControls } from "./DimensionControls";
import { ImageThumbnail } from "./ImageThumbnail";
import { SamplerSettings } from "./SamplerSettings";

// Reference image weight type options
const REFERENCE_WEIGHT_TYPE_OPTIONS: Record<string, string> = {
  linear: "linear",
  "ease in": "ease in",
  "ease out": "ease out",
  "ease in-out": "ease in-out",
  "reverse in-out": "reverse in-out",
  "weak input": "weak input",
  "weak output": "weak output",
  "weak middle": "weak middle",
  "strong middle": "strong middle",
  "style transfer": "style transfer",
  composition: "composition",
  "strong style transfer": "strong style transfer",
};

// Reference image combine mode options
const REFERENCE_COMBINE_MODE_OPTIONS: Record<string, string> = {
  concat: "concat",
  add: "add",
  subtract: "subtract",
  average: "average",
  "norm average": "norm average",
};

interface SortableReferenceThumbnailProps {
  id: string;
  filename: string;
  index: number;
  total: number;
  onRemove: () => void;
}

function SortableReferenceThumbnail({
  id,
  filename,
  index,
  total,
  onRemove,
}: SortableReferenceThumbnailProps) {
  const { attributes, listeners, setNodeRef, transform, transition } =
    useSortable({ id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <div ref={setNodeRef} style={style} {...attributes} {...listeners}>
      <ImageThumbnail
        filename={filename}
        context="reference"
        size="small"
        selectable={false}
        referenceIndex={index}
        referenceCount={total}
        onRemove={onRemove}
      />
    </div>
  );
}

interface ParameterPanelProps {
  // For Document Mode: pass documentId to generate assets instead of global images
  documentId?: string;
  // Optional override for generate action (Document Mode uses this)
  onGenerate?: () => void;
}

export function ParameterPanel({
  documentId,
  onGenerate,
}: ParameterPanelProps = {}) {
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const negativePromptRef = useRef<HTMLTextAreaElement>(null);

  const params = useImageStore((s) => s.params);
  const setParams = useImageStore((s) => s.setParams);
  const isGenerating = useImageStore((s) => s.isGenerating);
  const generationError = useImageStore((s) => s.generationError);
  const storeGenerate = useImageStore((s) => s.generate);

  // Use provided onGenerate or default to store generate with documentId
  const generate = onGenerate ?? (() => storeGenerate(documentId));
  const cancelGeneration = useImageStore((s) => s.cancelGeneration);
  const availableCheckpoints = useImageStore((s) => s.availableCheckpoints);
  const availableLoras = useImageStore((s) => s.availableLoras);
  const setCheckpoint = useImageStore((s) => s.setCheckpoint);
  const getCurrentArchitecture = useImageStore((s) => s.getCurrentArchitecture);
  const imageCount = useImageStore((s) => s.imageCount);
  const setImageCount = useImageStore((s) => s.setImageCount);
  const seedLocked = useImageStore((s) => s.ui.seedLocked);
  const setUi = useImageStore((s) => s.setUi);
  const removeReferenceImage = useImageStore((s) => s.removeReferenceImage);
  const reorderReferenceImages = useImageStore((s) => s.reorderReferenceImages);
  const setSourceImage = useImageStore((s) => s.setSourceImage);
  const addControlNet = useImageStore((s) => s.addControlNet);
  const updateControlNet = useImageStore((s) => s.updateControlNet);
  const removeControlNet = useImageStore((s) => s.removeControlNet);
  const preprocessControlNet = useImageStore((s) => s.preprocessControlNet);

  const { handleKeyDown: handlePromptKeyDown } = usePromptKeyboard({
    textareaRef: promptRef,
    value: params.prompt,
    onChange: (prompt) => setParams({ prompt }),
  });

  const { handleKeyDown: handleNegativePromptKeyDown } = usePromptKeyboard({
    textareaRef: negativePromptRef,
    value: params.negativePrompt,
    onChange: (negativePrompt) => setParams({ negativePrompt }),
  });

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5,
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      const referenceImages = params.referenceImages || [];
      const oldIndex = referenceImages.findIndex(
        (r) => r.filename === active.id
      );
      const newIndex = referenceImages.findIndex((r) => r.filename === over.id);
      if (oldIndex !== -1 && newIndex !== -1) {
        reorderReferenceImages(oldIndex, newIndex);
      }
    }
  };

  const currentArch = getCurrentArchitecture();

  const addLora = (filename: string) => {
    const loras = params.loras || [];
    if (!loras.some((l) => l.filename === filename)) {
      setParams({ loras: [...loras, { filename, weight: 1 }] });
    }
  };

  const getLoraTitle = (filename: string): string => {
    return (
      availableLoras.find((l) => l.filename === filename)?.title || filename
    );
  };

  const updateLoraWeight = (filename: string, weight: number) => {
    const loras = params.loras || [];
    setParams({
      loras: loras.map((l) => (l.filename === filename ? { ...l, weight } : l)),
    });
  };

  const removeLora = (filename: string) => {
    const loras = params.loras || [];
    setParams({ loras: loras.filter((l) => l.filename !== filename) });
  };

  // Filter LoRAs: unused and compatible with current model architecture
  const unusedLoras = availableLoras.filter((lora) => {
    // Already in use
    if ((params.loras || []).some((l) => l.filename === lora.filename)) {
      return false;
    }
    // If no model selected or architecture unknown, show all
    if (!currentArch || currentArch === "unknown") {
      return true;
    }
    // If LoRA architecture unknown, show it (user can decide)
    if (lora.architecture === "unknown") {
      return true;
    }
    // Only show compatible LoRAs
    return lora.architecture === currentArch;
  });

  // Group checkpoints by architecture for the dropdown
  const groupedCheckpoints = useMemo(() => {
    const groups: Record<string, typeof availableCheckpoints> = {};
    for (const model of availableCheckpoints) {
      const arch = model.architecture || "unknown";
      if (!groups[arch]) groups[arch] = [];
      groups[arch].push(model);
    }
    // Sort groups by architecture name, put "unknown" last
    return Object.entries(groups).sort(([a], [b]) => {
      if (a === "unknown") return 1;
      if (b === "unknown") return -1;
      return a.localeCompare(b);
    });
  }, [availableCheckpoints]);

  const canGenerate = params.model && params.prompt && !isGenerating;

  return (
    <Flex direction="column" height="100%" className="bg-[var(--gray-2)]">
      <Flex
        align="center"
        justify="between"
        px="4"
        py="3"
        className="border-b border-[var(--gray-6)]"
      >
        <Heading size="3">Image</Heading>
        <Flex gap="2" align="center">
          <Select.Root
            size="1"
            value={String(imageCount)}
            onValueChange={(value) => setImageCount(Number(value))}
          >
            <Select.Trigger style={{ width: 48 }} />
            <Select.Content>
              {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
                <Select.Item key={n} value={String(n)}>
                  {n}
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
          <Button
            size="1"
            variant="solid"
            disabled={!canGenerate}
            onClick={() => generate()}
          >
            {isGenerating ? "Generating..." : "Generate"}
          </Button>
          <IconButton
            size="1"
            variant="soft"
            color="red"
            disabled={!isGenerating}
            onClick={() => cancelGeneration()}
          >
            <StopCircleIcon size={16} weight="fill" />
          </IconButton>
        </Flex>
      </Flex>

      <ScrollArea scrollbars="vertical">
        <Flex direction="column" gap="3" p="4">
          {/* Error display */}
          {generationError && (
            <Callout.Root color="red" size="1">
              <Callout.Text>{generationError}</Callout.Text>
            </Callout.Root>
          )}

          {/* Prompt */}
          <Box>
            <Text as="label" size="2" weight="medium" mb="1">
              Prompt
            </Text>
            <TextArea
              ref={promptRef}
              className="h-40"
              placeholder="Enter your prompt..."
              value={params.prompt}
              onChange={(e) => setParams({ prompt: e.target.value })}
              onKeyDown={handlePromptKeyDown}
              rows={3}
            />
          </Box>

          {/* Negative Prompt */}
          <Box>
            <Text as="label" size="2" weight="medium" mb="1">
              Negative Prompt
            </Text>
            <TextArea
              ref={negativePromptRef}
              placeholder="What to avoid..."
              value={params.negativePrompt}
              onChange={(e) => setParams({ negativePrompt: e.target.value })}
              onKeyDown={handleNegativePromptKeyDown}
              rows={2}
            />
          </Box>

          {/* Seed */}
          <Flex align="center">
            <Text as="label" size="2" weight="medium" style={{ flex: 2 }}>
              Seed
            </Text>
            <Flex align="center" gap="2" style={{ flex: 3 }}>
              <TextField.Root
                size="2"
                style={{ flex: 1 }}
                type="number"
                value={params.seed}
                onChange={(e) =>
                  setParams({
                    seed: parseInt(e.target.value) || 1,
                  })
                }
              />
              <Switch
                checked={seedLocked}
                onCheckedChange={(checked) => setUi({ seedLocked: checked })}
              />
            </Flex>
          </Flex>

          {/* Divider */}
          <Box className="h-px bg-[var(--gray-5)]" />

          {/* Model */}
          <Flex align="center">
            <Text as="label" size="2" weight="medium" style={{ flex: 2 }}>
              Model
            </Text>
            <Box style={{ flex: 3, minWidth: 0 }}>
              <Select.Root
                size="2"
                value={params.model || ""}
                onValueChange={setCheckpoint}
              >
                <Select.Trigger
                  placeholder="Select model"
                  style={{ width: "100%", maxWidth: "100%" }}
                />
                <Select.Content>
                  {groupedCheckpoints.map(([arch, models]) => (
                    <Select.Group key={arch}>
                      <Select.Label className="flex items-center gap-2">
                        <span>{arch}</span>
                        <span className="h-px flex-1 bg-[var(--gray-6)]" />
                      </Select.Label>
                      {models.map((model) => (
                        <Select.Item
                          key={model.filename}
                          value={model.filename}
                        >
                          {model.title}
                        </Select.Item>
                      ))}
                    </Select.Group>
                  ))}
                </Select.Content>
              </Select.Root>
            </Box>
          </Flex>

          {/* Divider */}
          <Box className="h-px bg-[var(--gray-5)]" />

          {/* LoRA */}
          <Box>
            <Flex align="center">
              <Text as="label" size="2" weight="medium" style={{ flex: 2 }}>
                LoRA
              </Text>
              <Flex align="center" justify="between" style={{ flex: 3 }}>
                {(params.loras || []).length === 0 && (
                  <Text size="2" color="gray">
                    None
                  </Text>
                )}
                {(params.loras || []).length > 0 && <Box />}
                {unusedLoras.length > 0 && (
                  <DropdownMenu.Root>
                    <DropdownMenu.Trigger>
                      <IconButton size="1" variant="ghost">
                        <PlusIcon size={14} weight="bold" />
                      </IconButton>
                    </DropdownMenu.Trigger>
                    <DropdownMenu.Content>
                      {unusedLoras.map((lora) => (
                        <DropdownMenu.Item
                          key={lora.filename}
                          onSelect={() => addLora(lora.filename)}
                        >
                          {lora.title}
                        </DropdownMenu.Item>
                      ))}
                    </DropdownMenu.Content>
                  </DropdownMenu.Root>
                )}
              </Flex>
            </Flex>
            {(params.loras || []).map((lora) => (
              <Flex
                key={lora.filename}
                direction="column"
                gap="1"
                mt="2"
                p="2"
                className="rounded bg-[var(--gray-3)]"
              >
                <Flex justify="between" align="center">
                  <Text size="1" className="truncate" style={{ flex: 1 }}>
                    {getLoraTitle(lora.filename)}
                  </Text>
                  <Flex align="center" gap="2">
                    <Text size="1" color="gray">
                      {lora.weight.toFixed(2)}
                    </Text>
                    <IconButton
                      size="1"
                      variant="soft"
                      color="gray"
                      onClick={() => removeLora(lora.filename)}
                    >
                      <XIcon size={12} weight="bold" />
                    </IconButton>
                  </Flex>
                </Flex>
                <Slider
                  size="1"
                  value={[lora.weight]}
                  onValueChange={(value) => {
                    if (value[0] !== undefined) {
                      updateLoraWeight(lora.filename, value[0]);
                    }
                  }}
                  min={0}
                  max={2}
                  step={0.05}
                />
              </Flex>
            ))}
          </Box>

          {/* Source Image - only show when there is a source image */}
          {params.sourceImage && (
            <>
              {/* Divider */}
              <Box className="h-px bg-[var(--gray-5)]" />

              <Box>
                <Flex align="center">
                  <Text as="label" size="2" weight="medium" style={{ flex: 2 }}>
                    Source
                  </Text>
                  <Flex align="center" gap="2" style={{ flex: 3 }}>
                    <ImageThumbnail
                      filename={params.sourceImage}
                      context="source"
                      size="small"
                      selectable={false}
                      onRemove={() => setSourceImage("")}
                    />
                  </Flex>
                </Flex>
                <Flex align="center" gap="2" mt="2">
                  <Slider
                    size="1"
                    value={[params.sourceImageStrength]}
                    onValueChange={(value) => {
                      if (value[0] !== undefined) {
                        setParams({ sourceImageStrength: value[0] });
                      }
                    }}
                    min={0}
                    max={1}
                    step={0.05}
                    style={{ flex: 1 }}
                  />
                  <Text size="1" style={{ width: "2.5rem" }}>
                    {params.sourceImageStrength.toFixed(2)}
                  </Text>
                </Flex>
              </Box>
            </>
          )}

          {/* Reference - only show when there are reference images */}
          {(params.referenceImages || []).length > 0 && (
            <>
              {/* Divider */}
              <Box className="h-px bg-[var(--gray-5)]" />

              <Box>
                <Flex align="center">
                  <Text as="label" size="2" weight="medium" style={{ flex: 2 }}>
                    Reference
                  </Text>
                  <Box style={{ flex: 3 }}>
                    <DndContext
                      sensors={sensors}
                      collisionDetection={closestCenter}
                      onDragEnd={handleDragEnd}
                    >
                      <SortableContext
                        items={(params.referenceImages || []).map(
                          (r) => r.filename
                        )}
                        strategy={horizontalListSortingStrategy}
                      >
                        <Flex align="center" gap="2" wrap="wrap">
                          {(params.referenceImages || []).map((ref, index) => (
                            <SortableReferenceThumbnail
                              key={ref.filename}
                              id={ref.filename}
                              filename={ref.filename}
                              index={index}
                              total={(params.referenceImages || []).length}
                              onRemove={() =>
                                removeReferenceImage(ref.filename)
                              }
                            />
                          ))}
                        </Flex>
                      </SortableContext>
                    </DndContext>
                  </Box>
                </Flex>
                <Flex align="center" gap="2" mt="2">
                  <Slider
                    size="1"
                    value={[params.referenceWeight]}
                    onValueChange={(value) => {
                      if (value[0] !== undefined) {
                        setParams({ referenceWeight: value[0] });
                      }
                    }}
                    min={0}
                    max={2}
                    step={0.05}
                    style={{ flex: 1 }}
                  />
                  <Text size="1" style={{ width: "2.5rem" }}>
                    {params.referenceWeight.toFixed(2)}
                  </Text>
                  <Select.Root
                    size="1"
                    value={params.referenceWeightType}
                    onValueChange={(value) =>
                      setParams({
                        referenceWeightType:
                          value as typeof params.referenceWeightType,
                      })
                    }
                  >
                    <Select.Trigger />
                    <Select.Content>
                      {Object.entries(REFERENCE_WEIGHT_TYPE_OPTIONS).map(
                        ([value, label]) => (
                          <Select.Item key={value} value={value}>
                            {label}
                          </Select.Item>
                        )
                      )}
                    </Select.Content>
                  </Select.Root>
                  {(params.referenceImages || []).length > 1 && (
                    <Select.Root
                      size="1"
                      value={params.referenceCombineMode}
                      onValueChange={(value) =>
                        setParams({
                          referenceCombineMode:
                            value as typeof params.referenceCombineMode,
                        })
                      }
                    >
                      <Select.Trigger />
                      <Select.Content>
                        {Object.entries(REFERENCE_COMBINE_MODE_OPTIONS).map(
                          ([value, label]) => (
                            <Select.Item key={value} value={value}>
                              {label}
                            </Select.Item>
                          )
                        )}
                      </Select.Content>
                    </Select.Root>
                  )}
                </Flex>
              </Box>
            </>
          )}

          {/* ControlNet - show for SDXL and SD1.5 */}
          {(currentArch === "sdxl" || currentArch === "sd15") && (
            <>
              {/* Divider */}
              <Box className="h-px bg-[var(--gray-5)]" />

              <ControlNetSettings
                controlNets={params.controlNets || []}
                architecture={currentArch}
                onAdd={(type) => addControlNet("", type)}
                onUpdate={updateControlNet}
                onRemove={removeControlNet}
                onPreprocess={preprocessControlNet}
              />
            </>
          )}

          {/* Divider */}
          <Box className="h-px bg-[var(--gray-5)]" />

          {/* Dimensions */}
          <DimensionControls
            width={params.width}
            height={params.height}
            architecture={currentArch}
            onChange={(width, height) => setParams({ width, height })}
          />

          {/* Divider */}
          <Box className="h-px bg-[var(--gray-5)]" />

          {/* Sampler Settings */}
          <SamplerSettings
            steps={params.steps}
            cfgScale={params.cfgScale}
            sampler={params.sampler}
            scheduler={params.scheduler}
            architecture={currentArch}
            onChange={(updates) => setParams(updates)}
          />

          {/* Divider */}
          <Box className="h-px bg-[var(--gray-5)]" />

          {/* Face Detailer */}
          <Flex align="center">
            <Text as="label" size="2" weight="medium" style={{ flex: 2 }}>
              Face Detailer
            </Text>
            <Flex justify="end" style={{ flex: 3 }}>
              <Switch
                checked={params.faceDetailer}
                onCheckedChange={(checked) =>
                  setParams({ faceDetailer: checked })
                }
              />
            </Flex>
          </Flex>

          {/* Upscale */}
          <Flex align="center">
            <Text as="label" size="2" weight="medium" style={{ flex: 2 }}>
              Upscale
            </Text>
            <Flex align="center" gap="3" justify="end" style={{ flex: 3 }}>
              {params.upscaleEnabled && (
                <>
                  <Slider
                    size="1"
                    style={{ flex: 1 }}
                    value={[params.upscaleFactor]}
                    onValueChange={(value) =>
                      setParams({ upscaleFactor: value[0] })
                    }
                    min={1}
                    max={4}
                    step={1}
                  />
                  <Text size="2" color="gray" style={{ minWidth: 20 }}>
                    {params.upscaleFactor}x
                  </Text>
                </>
              )}
              <Switch
                checked={params.upscaleEnabled}
                onCheckedChange={(checked) =>
                  setParams({ upscaleEnabled: checked })
                }
              />
            </Flex>
          </Flex>
        </Flex>
      </ScrollArea>
    </Flex>
  );
}
