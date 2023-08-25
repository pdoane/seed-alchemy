import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config, utils
from .device import default_device, default_dtype
from .models import PromptGenRequest


class PromptGenerator:
    def __init__(self):
        self.device = default_device()
        self.torch_dtype = default_dtype()

    def __call__(self, req: PromptGenRequest):
        model_info = config.models[req.model]

        # Model
        repo_id = model_info.path

        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id)
        # model.to(self.device)

        # Input ids
        input_ids = tokenizer(req.prompt, return_tensors="pt").input_ids
        if input_ids.shape[1] == 0:
            input_ids = torch.asarray([[tokenizer.bos_token_id]], dtype=torch.long)
        input_ids = input_ids.repeat((req.count, 1))
        # input_ids = input_ids.to(self.device)

        # Generate
        utils.set_seed(req.seed)
        outputs = model.generate(
            input_ids,
            min_length=req.min_length,
            max_length=req.max_length,
            do_sample=True,
            num_beams=req.beam_count,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            length_penalty=req.length_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
