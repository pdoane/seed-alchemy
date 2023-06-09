from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DEISMultistepScheduler,
    DDPMScheduler,
)

DICT = {
    "ddim": (DDIMScheduler, {}),
    "ddpm": (DDPMScheduler, {}),
    "deis": (DEISMultistepScheduler, {}),
    "dpm++_2m_k": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "dpm++_2m_sde_k": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
    "dpm++_2m_sde": (DPMSolverMultistepScheduler, {"use_karras_sigmas": False, "algorithm_type": "sde-dpmsolver++"}),
    "dpm++_2m": (DPMSolverMultistepScheduler, {"use_karras_sigmas": False}),
    "dpm++_2s_k": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
    "dpm++_2s": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": False}),
    # Needs additional dependencies
    # "dpm++_sde_k": (DPMSolverSDEScheduler, {"use_karras_sigmas": True, "noise_sampler_seed": 0}),
    # "dpm++_sde": (DPMSolverSDEScheduler, {"use_karras_sigmas": False, "noise_sampler_seed": 0}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "euler_k": (EulerDiscreteScheduler, {"use_karras_sigmas": True}),
    "euler": (EulerDiscreteScheduler, {"use_karras_sigmas": False}),
    "heun_k": (HeunDiscreteScheduler, {"use_karras_sigmas": True}),
    "heun": (HeunDiscreteScheduler, {"use_karras_sigmas": False}),
    # Generates a black image
    # "kdpm_2_a": (KDPM2AncestralDiscreteScheduler, {}),
    # "kdpm_2": (KDPM2DiscreteScheduler, {}),
    "lms_k": (LMSDiscreteScheduler, {"use_karras_sigmas": True}),
    "lms": (LMSDiscreteScheduler, {"use_karras_sigmas": False}),
    "pndm": (PNDMScheduler, {}),
    "unipc": (UniPCMultistepScheduler, {}),
}
