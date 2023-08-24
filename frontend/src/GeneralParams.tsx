import { Fragment, useMemo } from "react";
import { useSnapshot } from "valtio";
import { CollapsibleContainer } from "./components/Container";
import { FormLabel } from "./components/FormLabel";
import { MenuGroup } from "./components/Menu";
import { Select, SelectItem } from "./components/Select";
import { SpinBox } from "./components/SpinBox";
import { useModels } from "./queries";
import { GeneralParamsState, GenerationParamsState } from "./schema";
import { toast } from "./components/Toast";

const schedulers = [
  "ddim",
  "ddpm",
  "deis",
  "dpm++_2m_k",
  "dpm++_2m_sde_k",
  "dpm++_2m_sde",
  "dpm++_2m",
  "dpm++_2s_k",
  "dpm++_2s",
  "euler_a",
  "euler_k",
  "euler",
  "heun_k",
  "heun",
  "lms_k",
  "lms",
  "pndm",
  "unipc",
];

interface GeneralParamsProps {
  state: GenerationParamsState;
}

interface GeneralParamProps {
  state: GeneralParamsState;
}

const ModelParam = ({ state }: GeneralParamsProps) => {
  const snapGeneral = useSnapshot(state.general);
  const queryModels = useModels();

  const data = useMemo(() => {
    const data: Record<string, string[]> = {};
    queryModels.data?.forEach(([type, base, model]) => {
      if (type === "checkpoint") {
        if (!data[base]) {
          data[base] = [];
        }
        data[base].push(model);
      }
    });

    return data;
  }, [queryModels.data]);

  function getBaseModelType(model: string): string | undefined {
    return queryModels.data?.find(([_, __, m]) => m === model)?.[1];
  }

  function handleChange(value: string): void {
    state.general.model = value;
    const baseModelType = getBaseModelType(value);

    let modelsCleared = 0;
    for (let i = state.lora.entries.length - 1; i >= 0; i--) {
      const entry = state.lora.entries[i];
      if (getBaseModelType(entry.model) != baseModelType) {
        state.lora.entries.splice(i, 1);
        ++modelsCleared;
      }
    }

    for (let i = state.controlNet.conditions.length - 1; i >= 0; i--) {
      const condition = state.controlNet.conditions[i];
      if (getBaseModelType(condition.model) != baseModelType) {
        state.controlNet.conditions.splice(i, 1);
        ++modelsCleared;
      }
    }

    if (modelsCleared > 0)
      toast("Base model changed. Cleared " + modelsCleared.toString() + " incompatible submodel(s)");
  }

  return (
    <FormLabel label="Stable Diffusion Model">
      <Select value={snapGeneral.model} onChange={handleChange} status={queryModels.status}>
        {Object.entries(data).map(([base, models]) => (
          <Fragment key={base}>
            <MenuGroup label={base} />
            {models.map((model) => (
              <SelectItem key={model} text={model} value={model} />
            ))}
          </Fragment>
        ))}
      </Select>
    </FormLabel>
  );
};

const SchedulerParam = ({ state }: GeneralParamProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Scheduler">
      <Select value={snap.scheduler} onChange={(x) => (state.scheduler = x)}>
        {schedulers.map((str, _) => (
          <SelectItem key={str} text={str} value={str} />
        ))}
      </Select>
    </FormLabel>
  );
};

const ImageCountParam = ({ state }: GeneralParamProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Images">
      <SpinBox value={snap.imageCount} onChange={(x) => (state.imageCount = x)} min={1} max={100} />
    </FormLabel>
  );
};

const StepsParam = ({ state }: GeneralParamProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Steps">
      <SpinBox value={snap.steps} onChange={(x) => (state.steps = x)} min={1} max={100} />
    </FormLabel>
  );
};

const CfgScaleParam = ({ state }: GeneralParamProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="CFG Scale">
      <SpinBox value={snap.cfgScale} onChange={(x) => (state.cfgScale = x)} min={1.0} max={200} step={0.5} />
    </FormLabel>
  );
};

const WidthParam = ({ state }: GeneralParamProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Width">
      <SpinBox value={snap.width} onChange={(x) => (state.width = x)} min={64} max={2048} step={64} />
    </FormLabel>
  );
};

const HeightParam = ({ state }: GeneralParamProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Height">
      <SpinBox value={snap.height} onChange={(x) => (state.height = x)} min={64} max={2048} step={64} />
    </FormLabel>
  );
};

export const GeneralParams = ({ state }: GeneralParamsProps) => {
  const snapGeneral = useSnapshot(state.general);

  return (
    <CollapsibleContainer
      label="General"
      isOpen={snapGeneral.isOpen}
      onIsOpenChange={(x) => (state.general.isOpen = x)}
    >
      <ModelParam state={state} />
      <SchedulerParam state={state.general} />
      <div className="flex space-x-3">
        <ImageCountParam state={state.general} />
        <StepsParam state={state.general} />
        <CfgScaleParam state={state.general} />
      </div>
      <div className="flex space-x-3">
        <WidthParam state={state.general} />
        <HeightParam state={state.general} />
      </div>
    </CollapsibleContainer>
  );
};
