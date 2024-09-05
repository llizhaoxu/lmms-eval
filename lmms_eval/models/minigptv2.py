import torch
import logging
import copy
from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple

from .minigpt_model.minigpt_v2 import MiniGPTv2
from .minigpt_model.blip_processors import  Blip2ImageEvalProcessor,BlipCaptionProcessor

from lmms_eval.utils import stop_sequences_criteria


import warnings

warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")


@register_model("minigpt_v2")
class Minigpt_v2(lmms):
    """
    InstructBLIP Model
    """

    def __init__(
        self,
        pretrained: str = "Salesforce/instructblip-vicuna-7b",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device
        config={
            "llama_model": "/home/li0007xu/DTC/llama-chat",
            "ckpt": "/home/li0007xu/DTC/Minigpt_ckpt/minigptv2_checkpoint.pth",
            "image_size": 448,
            "drop_path_rate": 0,
            "use_grad_checkpoint": False,
            "vit_precision": "fp16",
            "freeze_vit": True,
            "prompt": "",
            "lora_r": 64,
            "lora_alpha": 16,
            "end_sym": "</s>",
            "prompt_template": '[INST] {} [/INST]',
            "max_txt_len": 500,
        }
        self._model = MiniGPTv2.from_config(config)
        self._image_processor = Blip2ImageEvalProcessor(image_size=448)



        self.model.eval()

        self.batch_size_per_gpu = int(batch_size)
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1





    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model



    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size



    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "We have not implemented this function for InstructBLIP yet"

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")


        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            context = contexts
            if task=="coco2014_cap_val":
                template = "<s>[INST] <Img><ImageHere></Img> [grounding] <question> [/INST]:"
            else:
                template = "<s>[INST] <Img><ImageHere></Img> <question> [/INST]:"
            context = template.replace("<question>", context)
            visual=self._image_processor(visuals[0])
            
            visual=visual.unsqueeze(0)

            gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            try:
                cont = self.model.generate(
                    visual,
                    [context],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                cont = ""
            cont=  cont[0].split('###')[0]  # remove the stop sign '###'
            cont = cont.split('Assistant:')[-1].strip()

            res.append(cont)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), cont)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        

        pbar.close()
        return res
