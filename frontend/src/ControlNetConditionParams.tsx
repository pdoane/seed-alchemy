import { Fragment } from "react";
import { AiFillEye, AiFillThunderbolt } from "react-icons/ai";
import { useSnapshot } from "valtio";
import { SourceImage } from "./SourceImage";
import { IconButton } from "./components/Button";
import { FormLabel } from "./components/FormLabel";
import { MenuGroup } from "./components/Menu";
import { Select, SelectItem } from "./components/Select";
import { Slider } from "./components/Slider";
import { usePreviewProcessor } from "./mutations";
import { useControlNetModels } from "./queries";
import { ControlNetConditionParamsState } from "./schema";

const categoriesToProcessors: { [key: string]: string[] } = {
  none: ["none"],
  canny: ["canny"],
  depth: ["depth_leres", "depth_leres++", "depth_midas", "depth_zoe"],
  lineart: ["lineart_anime", "lineart_coarse", "lineart_realistic"],
  mediapipe: ["mediapipe_face"],
  mlsd: ["mlsd"],
  normal: ["normal_bae", "normal_midas"],
  openpose: ["openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand"],
  scribble: ["scribble_hed", "scribble_hedsafe", "scribble_pidinet", "scribble_pidisafe", "scribble_xdog"],
  shuffle: ["shuffle"],
  softedge: ["softedge_hed", "softedge_hedsafe", "softedge_pidinet", "softedge_pidsafe"],
};

const processorsToModels: { [key: string]: string[] } = {
  none: [],
  canny: ["control_v11p_sd15_canny", "control_sd15_canny"],
  depth_leres: ["control_v11f1p_sd15_depth"],
  "depth_leres++": ["control_v11f1p_sd15_depth"],
  depth_midas: ["control_v11f1p_sd15_depth", "control_sd15_depth"],
  depth_zoe: ["control_v11f1p_sd15_depth"],
  lineart_anime: ["control_v11p_sd15s2_lineart_anime"],
  lineart_coarse: ["control_v11p_sd15_lineart"],
  lineart_realistic: ["control_v11p_sd15_lineart"],
  mediapipe_face: ["control_v2p_sd15_mediapipe_face", "control_v2p_sd21_mediapipe_face"],
  mlsd: ["control_v11p_sd15_mlsd", "control_sd15_mlsd"],
  normal_bae: ["control_v11p_sd15_normalbae"],
  normal_midas: ["control_sd15_normal"],
  openpose: ["control_v11p_sd15_openpose", "control_sd15_openpose"],
  openpose_face: ["control_v11p_sd15_openpose"],
  openpose_faceonly: ["control_v11p_sd15_openpose"],
  openpose_full: ["control_v11p_sd15_openpose"],
  openpose_hand: ["control_v11p_sd15_openpose"],
  scribble_hed: ["control_v11p_sd15_scribble", "control_sd15_scribble"],
  scribble_hedsafe: ["control_v11p_sd15_scribble", "control_sd15_scribble"],
  scribble_pidinet: ["control_v11p_sd15_scribble", "control_sd15_scribble"],
  scribble_pidisafe: ["control_v11p_sd15_scribble", "control_sd15_scribble"],
  scribble_xdog: ["control_v11p_sd15_scribble", "control_sd15_scribble"],
  shuffle: ["control_v11e_sd15_shuffle"],
  softedge_hed: ["control_v11p_sd15_softedge", "control_sd15_hed"],
  softedge_hedsafe: ["control_v11p_sd15_softedge", "control_sd15_hed"],
  softedge_pidinet: ["control_v11p_sd15_softedge", "control_sd15_hed"],
  softedge_pidsafe: ["control_v11p_sd15_softedge", "control_sd15_hed"],
};

interface ControlNetConditionParamsProps {
  state: ControlNetConditionParamsState;
  baseModelType: string | undefined;
}

const ControlNetProcessorParam = ({ state, baseModelType }: ControlNetConditionParamsProps) => {
  const snap = useSnapshot(state);
  const queryControlNetModels = useControlNetModels(baseModelType);
  const postPreviewProcessor = usePreviewProcessor(state);

  function handleSyncModel(): void {
    const models = processorsToModels[state.processor];
    const model = models.find((value) => queryControlNetModels.data?.includes(value));
    if (model) {
      state.model = model;
    }
  }

  function handlePreview(): void {
    postPreviewProcessor.mutate();
  }

  return (
    <FormLabel label="Processor">
      <div className="flex space-x-1">
        <Select value={snap.processor} onChange={(x) => (state.processor = x)}>
          {Object.keys(categoriesToProcessors).map((category) => (
            <Fragment key={category}>
              <MenuGroup label={category} />
              {categoriesToProcessors[category].map((str) => (
                <SelectItem key={str} text={str} value={str} />
              ))}
            </Fragment>
          ))}
        </Select>
        <IconButton icon={AiFillThunderbolt} onClick={handleSyncModel} />
        <IconButton icon={AiFillEye} onClick={handlePreview} disabled={postPreviewProcessor.isLoading} />
      </div>
    </FormLabel>
  );
};

const ControlNetModelParam = ({ state, baseModelType }: ControlNetConditionParamsProps) => {
  const snap = useSnapshot(state);
  const queryControlNetModels = useControlNetModels(baseModelType);

  return (
    <FormLabel label="Model">
      <Select value={snap.model} onChange={(x) => (state.model = x)} status={queryControlNetModels.status}>
        {queryControlNetModels.data?.map((str, _) => (
          <SelectItem key={str} text={str} value={str} />
        ))}
      </Select>
    </FormLabel>
  );
};

const ControlNetSourceParam = ({ state }: ControlNetConditionParamsProps) => {
  const snap = useSnapshot(state, { sync: true });

  return <SourceImage label="Source" value={snap.source} onChange={(x) => (state.source = x)} />;
};

const ControlNetScaleParam = ({ state }: ControlNetConditionParamsProps) => {
  const snap = useSnapshot(state);

  return (
    <FormLabel label="Scale">
      <Slider value={snap.scale} onChange={(x) => (state.scale = x)} max={2.0} />
    </FormLabel>
  );
};

export const ControlNetConditionParams = (props: ControlNetConditionParamsProps) => {
  return (
    <div className="w-full space-y-2">
      <ControlNetProcessorParam {...props} />
      <ControlNetModelParam {...props} />
      <ControlNetSourceParam {...props} />
      <ControlNetScaleParam {...props} />
    </div>
  );
};
