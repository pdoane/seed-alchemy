import {
  Box,
  Flex,
  IconButton,
  Select,
  Slider,
  Text,
  Tooltip,
} from "@radix-ui/themes";
import { FadersHorizontalIcon } from "@phosphor-icons/react";
import type { Architecture } from "../../shared/types/models";

// ComfyUI sampler names with display labels
const SAMPLER_OPTIONS: Record<string, string> = {
  euler: "Euler",
  euler_ancestral: "Euler A",
  dpmpp_2m: "DPM++ 2M",
  res_multistep: "Res Multi",
};

// ComfyUI scheduler (noise schedule) options
const SCHEDULER_OPTIONS: Record<string, string> = {
  normal: "Normal",
  karras: "Karras",
  simple: "Simple",
};

// Architecture-specific recommended defaults
const ARCHITECTURE_DEFAULTS: Record<
  string,
  { steps: number; cfgScale: number; sampler: string; scheduler: string }
> = {
  sd15: { steps: 20, cfgScale: 7, sampler: "euler", scheduler: "normal" },
  sd20: { steps: 20, cfgScale: 7, sampler: "euler", scheduler: "normal" },
  sd21: { steps: 20, cfgScale: 7, sampler: "euler", scheduler: "normal" },
  sdxl: { steps: 25, cfgScale: 5, sampler: "euler", scheduler: "normal" },
  "sdxl-refiner": {
    steps: 25,
    cfgScale: 5,
    sampler: "euler",
    scheduler: "normal",
  },
  sd3: { steps: 28, cfgScale: 4.5, sampler: "euler", scheduler: "simple" },
  sd35: { steps: 28, cfgScale: 4.5, sampler: "euler", scheduler: "simple" },
  flux: { steps: 20, cfgScale: 1, sampler: "euler", scheduler: "simple" },
  zit: { steps: 4, cfgScale: 1, sampler: "res_multistep", scheduler: "simple" },
  cascade: { steps: 20, cfgScale: 4, sampler: "euler", scheduler: "normal" },
  wan: { steps: 20, cfgScale: 7, sampler: "euler", scheduler: "normal" },
};

interface SamplerSettingsProps {
  steps: number;
  cfgScale: number;
  sampler: string;
  scheduler: string;
  architecture: Architecture | null;
  onChange: (
    updates: Partial<{
      steps: number;
      cfgScale: number;
      sampler: string;
      scheduler: string;
    }>
  ) => void;
}

export function SamplerSettings({
  steps,
  cfgScale,
  sampler,
  scheduler,
  architecture,
  onChange,
}: SamplerSettingsProps) {
  const defaults = architecture ? ARCHITECTURE_DEFAULTS[architecture] : null;

  const applyDefaults = () => {
    if (defaults) {
      onChange(defaults);
    }
  };

  return (
    <Box>
      {/* Header row with label and defaults button */}
      <Flex align="center" justify="between" mb="2">
        <Text as="label" size="2" weight="medium">
          Sampler
        </Text>
        {defaults && (
          <Tooltip content="Apply recommended settings for this model">
            <IconButton variant="ghost" size="1" onClick={applyDefaults}>
              <FadersHorizontalIcon size={14} />
            </IconButton>
          </Tooltip>
        )}
      </Flex>

      {/* Row 1: Steps slider and CFG slider */}
      <Flex align="center" gap="2">
        <Text size="1" color="gray" style={{ width: "2.5rem" }}>
          Steps
        </Text>
        <Slider
          size="1"
          style={{ flex: 1 }}
          value={[steps]}
          onValueChange={(value) => onChange({ steps: value[0] })}
          min={1}
          max={75}
          step={1}
        />
        <Text size="1" style={{ width: "1.5rem", textAlign: "right" }}>
          {steps}
        </Text>
        <Text
          size="1"
          color="gray"
          style={{ width: "2rem", marginLeft: "4px" }}
        >
          CFG
        </Text>
        <Slider
          size="1"
          style={{ flex: 1 }}
          value={[cfgScale]}
          onValueChange={(value) => onChange({ cfgScale: value[0] })}
          min={1}
          max={15}
          step={0.5}
        />
        <Text size="1" style={{ width: "2rem", textAlign: "right" }}>
          {cfgScale.toFixed(1)}
        </Text>
      </Flex>

      {/* Row 2: Sampler and Scheduler selects */}
      <Flex align="center" gap="2" mt="2">
        <Select.Root
          size="1"
          value={sampler}
          onValueChange={(value) => onChange({ sampler: value })}
        >
          <Select.Trigger style={{ flex: 1 }} />
          <Select.Content>
            {Object.entries(SAMPLER_OPTIONS).map(([value, label]) => (
              <Select.Item key={value} value={value}>
                {label}
              </Select.Item>
            ))}
          </Select.Content>
        </Select.Root>
        <Select.Root
          size="1"
          value={scheduler}
          onValueChange={(value) => onChange({ scheduler: value })}
        >
          <Select.Trigger style={{ flex: 1 }} />
          <Select.Content>
            {Object.entries(SCHEDULER_OPTIONS).map(([value, label]) => (
              <Select.Item key={value} value={value}>
                {label}
              </Select.Item>
            ))}
          </Select.Content>
        </Select.Root>
      </Flex>
    </Box>
  );
}
