from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from controlnet_aux import (
    CannyDetector,
    ContentShuffleDetector,
    HEDdetector,
    LeresDetector,
    LineartAnimeDetector,
    LineartDetector,
    MediapipeFaceDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
    OpenposeDetector,
    PidiNetDetector,
    ZoeDetector,
)
from PIL import Image

from .detectors import ScribbleXDoGDetector
from .types import ModelType, BaseModelType


@dataclass
class ProcessorInfo:
    cls: type
    repo_id: Optional[str] = None
    params: dict[str, bool] = field(default_factory=dict)
    post_process: Optional[Callable[[Any], Image.Image]] = None


processors = {
    "none": ProcessorInfo(
        cls=None,
    ),
    "canny": ProcessorInfo(
        cls=CannyDetector,
    ),
    "depth_leres": ProcessorInfo(
        cls=LeresDetector,
        repo_id="lllyasviel/Annotators",
        params={"boost": False},
    ),
    "depth_leres++": ProcessorInfo(
        cls=LeresDetector,
        repo_id="lllyasviel/Annotators",
        params={"boost": True},
    ),
    "depth_midas": ProcessorInfo(
        cls=MidasDetector,
        repo_id="lllyasviel/Annotators",
    ),
    "depth_zoe": ProcessorInfo(
        cls=ZoeDetector,
        repo_id="lllyasviel/Annotators",
    ),
    "lineart_anime": ProcessorInfo(
        cls=LineartAnimeDetector,
        repo_id="lllyasviel/Annotators",
    ),
    "lineart_coarse": ProcessorInfo(
        cls=LineartDetector,
        repo_id="lllyasviel/Annotators",
        params={"coarse": True},
    ),
    "lineart_realistic": ProcessorInfo(
        cls=LineartDetector,
        repo_id="lllyasviel/Annotators",
        params={"coarse": False},
    ),
    "mediapipe_face": ProcessorInfo(
        cls=MediapipeFaceDetector,
    ),
    "mlsd": ProcessorInfo(
        cls=MLSDdetector,
        repo_id="lllyasviel/Annotators",
    ),
    "normal_bae": ProcessorInfo(
        cls=NormalBaeDetector,
        repo_id="lllyasviel/Annotators",
    ),
    "normal_midas": ProcessorInfo(
        cls=MidasDetector,
        repo_id="lllyasviel/Annotators",
        params={"depth_and_normal": True},
        post_process=lambda images: images[1],
    ),
    "openpose_face": ProcessorInfo(
        cls=OpenposeDetector,
        repo_id="lllyasviel/Annotators",
        params={"include_body": True, "include_hand": False, "include_face": True},
    ),
    "openpose_faceonly": ProcessorInfo(
        cls=OpenposeDetector,
        repo_id="lllyasviel/Annotators",
        params={"include_body": False, "include_hand": False, "include_face": True},
    ),
    "openpose_full": ProcessorInfo(
        cls=OpenposeDetector,
        repo_id="lllyasviel/Annotators",
        params={"include_body": True, "include_hand": True, "include_face": True},
    ),
    "openpose_hand": ProcessorInfo(
        cls=OpenposeDetector,
        repo_id="lllyasviel/Annotators",
        params={"include_body": False, "include_hand": True, "include_face": False},
    ),
    "openpose": ProcessorInfo(
        cls=OpenposeDetector,
        repo_id="lllyasviel/Annotators",
        params={"include_body": True, "include_hand": False, "include_face": False},
    ),
    "scribble_hed": ProcessorInfo(
        cls=HEDdetector,
        repo_id="lllyasviel/Annotators",
        params={"scribble": True},
    ),
    "scribble_hedsafe": ProcessorInfo(
        cls=HEDdetector,
        repo_id="lllyasviel/Annotators",
        params={"scribble": True, "safe": True},
    ),
    "scribble_pidinet": ProcessorInfo(
        cls=PidiNetDetector,
        repo_id="lllyasviel/Annotators",
        params={"safe": False, "scribble": True},
    ),
    "scribble_pidisafe": ProcessorInfo(
        cls=PidiNetDetector,
        repo_id="lllyasviel/Annotators",
        params={"safe": True, "scribble": True},
    ),
    "scribble_xdog": ProcessorInfo(
        cls=ScribbleXDoGDetector,
    ),
    "shuffle": ProcessorInfo(
        cls=ContentShuffleDetector,
    ),
    "softedge_hed": ProcessorInfo(
        cls=HEDdetector,
        repo_id="lllyasviel/Annotators",
        params={"scribble": False, "safe": False},
    ),
    "softedge_hedsafe": ProcessorInfo(
        cls=HEDdetector,
        repo_id="lllyasviel/Annotators",
        params={"scribble": False, "safe": True},
    ),
    "softedge_pidinet": ProcessorInfo(
        cls=PidiNetDetector,
        repo_id="lllyasviel/Annotators",
        params={"safe": False, "scribble": False},
    ),
    "softedge_pidsafe": ProcessorInfo(
        cls=PidiNetDetector,
        repo_id="lllyasviel/Annotators",
        params={"safe": True, "scribble": False},
    ),
}


v10_models: list[(ModelType, BaseModelType, str, str)] = [
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/sd-controlnet-canny", "control_sd15_canny"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/sd-controlnet-depth", "control_sd15_depth"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/sd-controlnet-hed", "control_sd15_hed"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/sd-controlnet-mlsd", "control_sd15_mlsd"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/sd-controlnet-normal", "control_sd15_normal"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/sd-controlnet-openpose", "control_sd15_openpose"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/sd-controlnet-scribble", "control_sd15_scribble"),
    # (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/sd-controlnet-seg", "control_sd15_seg"),
]

v11_models: list[(str, ModelType, BaseModelType, str)] = [
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/control_v11p_sd15_canny", "control_v11p_sd15_canny"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/control_v11f1p_sd15_depth", "control_v11f1p_sd15_depth"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/control_v11p_sd15_inpaint", "control_v11p_sd15_inpaint"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/control_v11e_sd15_ip2p", "control_v11e_sd15_ip2p"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/control_v11p_sd15_lineart", "control_v11p_sd15_lineart"),
    (
        ModelType.ControlNet,
        BaseModelType.SD_1,
        "lllyasviel/control_v11p_sd15s2_lineart_anime",
        "control_v11p_sd15s2_lineart_anime",
    ),
    (
        ModelType.ControlNet,
        BaseModelType.SD_1,
        "lllyasviel/control_v11p_sd15_mlsd",
        "control_v11p_sd15_mlsd",
    ),
    (
        ModelType.ControlNet,
        BaseModelType.SD_1,
        "lllyasviel/control_v11p_sd15_normalbae",
        "control_v11p_sd15_normalbae",
    ),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/control_v11p_sd15_openpose", "control_v11p_sd15_openpose"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/control_v11p_sd15_scribble", "control_v11p_sd15_scribble"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/control_v11p_sd15_seg", "control_v11p_sd15_seg"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/control_v11e_sd15_shuffle", "control_v11e_sd15_shuffle"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/control_v11p_sd15_softedge", "control_v11p_sd15_softedge"),
    (ModelType.ControlNet, BaseModelType.SD_1, "lllyasviel/control_v11u_sd15_tile", "control_v11u_sd15_tile"),
]

mediapipe_v2_models: list[(str, ModelType, BaseModelType, str)] = {
    (
        ModelType.ControlNet,
        BaseModelType.SD_1,
        "CrucibleAI/ControlNetMediaPipeFace/diffusion_sd15",
        "control_v2p_sd15_mediapipe_face",
    ),
    (
        ModelType.ControlNet,
        BaseModelType.SD_2,
        "CrucibleAI/ControlNetMediaPipeFace",
        "control_v2p_sd21_mediapipe_face",
    ),
}
