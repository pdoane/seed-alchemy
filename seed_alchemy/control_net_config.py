from dataclasses import dataclass

from PySide6.QtCore import QSettings

from .processors import (
    CannyProcessor,
    DepthLeresBoostProcessor,
    DepthLeresProcessor,
    DepthMidasProcessor,
    DepthZoeProcessor,
    LineartAnimeProcessor,
    LineartCoarseProcessor,
    LineartProcessor,
    MediapipeFaceProcessor,
    MlsdProcessor,
    NormalBaeProcessor,
    NormalMidasProcessor,
    OpenposeFaceOnlyProcessor,
    OpenposeFaceProcessor,
    OpenposeFullProcessor,
    OpenposeHandProcessor,
    OpenposeProcessor,
    ProcessorBase,
    ScribbleHEDProcessor,
    ScribblePIDIProcessor,
    ScribbleXDoGProcessor,
    SegProcessor,
    ShuffleProcessor,
    SoftEdgeHEDProcessor,
    SoftEdgeHEDSafeProcessor,
    SoftEdgePIDIProcessor,
    SoftEdgePIDISafeProcessor,
)

preprocessors: dict[str, ProcessorBase] = {
    "none": None,
    "canny": CannyProcessor,
    "depth_leres": DepthLeresProcessor,
    "depth_leres++": DepthLeresBoostProcessor,
    "depth_midas": DepthMidasProcessor,
    "depth_zoe": DepthZoeProcessor,
    "lineart_anime": LineartAnimeProcessor,
    "lineart_coarse": LineartCoarseProcessor,
    "lineart_realistic": LineartProcessor,
    "mediapipe_face": MediapipeFaceProcessor,
    "mlsd": MlsdProcessor,
    "normal_bae": NormalBaeProcessor,
    "normal_midas": NormalMidasProcessor,
    "openpose": OpenposeProcessor,
    "openpose_face": OpenposeFaceProcessor,
    "openpose_faceonly": OpenposeFaceOnlyProcessor,
    "openpose_full": OpenposeFullProcessor,
    "openpose_hand": OpenposeHandProcessor,
    "scribble_hed": ScribbleHEDProcessor,
    "scribble_pidinet": ScribblePIDIProcessor,
    "scribble_xdog": ScribbleXDoGProcessor,
    "seg_ofade20k": SegProcessor,
    "shuffle": ShuffleProcessor,
    "softedge_hed": SoftEdgeHEDProcessor,
    "softedge_hedsafe": SoftEdgeHEDSafeProcessor,
    "softedge_pidinet": SoftEdgePIDIProcessor,
    "softedge_pidsafe": SoftEdgePIDISafeProcessor,
}

preprocessors_to_models: dict[str, list[str]] = {
    "none": [],
    "canny": ["control_v11p_sd15_canny", "control_sd15_canny"],
    "depth_leres": ["control_v11f1p_sd15_depth"],
    "depth_leres++": ["control_v11f1p_sd15_depth"],
    "depth_midas": ["control_v11f1p_sd15_depth", "control_sd15_depth"],
    "depth_zoe": ["control_v11f1p_sd15_depth"],
    "lineart_anime": ["control_v11p_sd15s2_lineart_anime"],
    "lineart_coarse": ["control_v11p_sd15_lineart"],
    "lineart_realistic": ["control_v11p_sd15_lineart"],
    "mediapipe_face": ["control_v2p_sd15_mediapipe_face"],
    "mlsd": ["control_v11p_sd15_mlsd", "control_sd15_mlsd"],
    "normal_bae": ["control_v11p_sd15_normalbae"],
    "normal_midas": ["control_sd15_normal"],
    "openpose": ["control_v11p_sd15_openpose", "control_sd15_openpose"],
    "openpose_face": ["control_v11p_sd15_openpose"],
    "openpose_faceonly": ["control_v11p_sd15_openpose"],
    "openpose_full": ["control_v11p_sd15_openpose"],
    "openpose_hand": ["control_v11p_sd15_openpose"],
    "scribble_hed": ["control_v11p_sd15_scribble", "control_sd15_scribble"],
    "scribble_pidinet": ["control_v11p_sd15_scribble", "control_sd15_scribble"],
    "scribble_xdog": ["control_v11p_sd15_scribble", "control_sd15_scribble"],
    "seg_ofade20k": ["control_v11p_sd15_seg", "control_sd15_seg"],
    "shuffle": ["control_v11e_sd15_shuffle"],
    "softedge_hed": ["control_v11p_sd15_softedge", "control_sd15_hed"],
    "softedge_hedsafe": ["control_v11p_sd15_softedge", "control_sd15_hed"],
    "softedge_pidinet": ["control_v11p_sd15_softedge", "control_sd15_hed"],
    "softedge_pidsafe": ["control_v11p_sd15_softedge", "control_sd15_hed"],
}


@dataclass
class ControlNetModel:
    repo_id: str
    subfolder: str = None


models: dict[str, ControlNetModel] = {
    "control_sd15_canny": ControlNetModel(repo_id="lllyasviel/sd-controlnet-canny"),
    "control_sd15_depth": ControlNetModel(repo_id="lllyasviel/sd-controlnet-depth"),
    "control_sd15_hed": ControlNetModel(repo_id="lllyasviel/sd-controlnet-hed"),
    "control_sd15_mlsd": ControlNetModel(repo_id="lllyasviel/sd-controlnet-mlsd"),
    "control_sd15_normal": ControlNetModel(repo_id="lllyasviel/sd-controlnet-normal"),
    "control_sd15_openpose": ControlNetModel(repo_id="lllyasviel/sd-controlnet-openpose"),
    "control_sd15_scribble": ControlNetModel(repo_id="lllyasviel/sd-controlnet-scribble"),
    "control_sd15_seg": ControlNetModel(repo_id="lllyasviel/sd-controlnet-seg"),
    "control_v11p_sd15_canny": ControlNetModel(repo_id="lllyasviel/control_v11p_sd15_canny"),
    "control_v11f1p_sd15_depth": ControlNetModel(repo_id="lllyasviel/control_v11f1p_sd15_depth"),
    #'control_v11p_sd15_inpaint': ControlNetModel(repo_id='lllyasviel/control_v11p_sd15_inpaint'),
    "control_v11e_sd15_ip2p": ControlNetModel(repo_id="lllyasviel/control_v11e_sd15_ip2p"),
    "control_v11p_sd15_lineart": ControlNetModel(repo_id="lllyasviel/control_v11p_sd15_lineart"),
    "control_v11p_sd15s2_lineart_anime": ControlNetModel(repo_id="lllyasviel/control_v11p_sd15s2_lineart_anime"),
    "control_v11p_sd15_mlsd": ControlNetModel(repo_id="lllyasviel/control_v11p_sd15_mlsd"),
    "control_v11p_sd15_normalbae": ControlNetModel(repo_id="lllyasviel/control_v11p_sd15_normalbae"),
    "control_v11p_sd15_openpose": ControlNetModel(repo_id="lllyasviel/control_v11p_sd15_openpose"),
    "control_v11p_sd15_scribble": ControlNetModel(repo_id="lllyasviel/control_v11p_sd15_scribble"),
    "control_v11p_sd15_seg": ControlNetModel(repo_id="lllyasviel/control_v11p_sd15_seg"),
    "control_v11e_sd15_shuffle": ControlNetModel(repo_id="lllyasviel/control_v11e_sd15_shuffle"),
    "control_v11p_sd15_softedge": ControlNetModel(repo_id="lllyasviel/control_v11p_sd15_softedge"),
    #'control_v11u_sd15_tile': ControlNetModel(repo_id='lllyasviel/control_v11u_sd15_tile'),
    "control_v2p_sd15_mediapipe_face": ControlNetModel(
        repo_id="CrucibleAI/ControlNetMediaPipeFace", subfolder="diffusion_sd15"
    ),
}

v10_models: list[str] = [
    "control_sd15_canny",
    "control_sd15_depth",
    "control_sd15_hed",
    "control_sd15_mlsd",
    "control_sd15_normal",
    "control_sd15_openpose",
    "control_sd15_scribble",
    "control_sd15_seg",
]

v11_models: list[str] = [
    "control_v11p_sd15_canny",
    "control_v11f1p_sd15_depth",
    #'control_v11p_sd15_inpaint',
    "control_v11e_sd15_ip2p",
    "control_v11p_sd15_lineart",
    "control_v11p_sd15s2_lineart_anime",
    "control_v11p_sd15_mlsd",
    "control_v11p_sd15_normalbae",
    "control_v11p_sd15_openpose",
    "control_v11p_sd15_scribble",
    "control_v11p_sd15_seg",
    "control_v11e_sd15_shuffle",
    "control_v11p_sd15_softedge",
    #'control_v11u_sd15_tile',
]

mediapipe_v2_models: list[str] = {"control_v2p_sd15_mediapipe_face"}

installed_models: list[str] = []


def load_from_settings(settings: QSettings):
    global installed_models
    installed_models = []
    if settings.value("install_control_net_v10", type=bool):
        installed_models += v10_models
    if settings.value("install_control_net_v11", type=bool):
        installed_models += v11_models
    if settings.value("install_control_net_mediapipe_v2", type=bool):
        installed_models += mediapipe_v2_models
    installed_models = sorted(installed_models)
