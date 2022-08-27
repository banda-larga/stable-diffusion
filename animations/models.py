import torch
from omegaconf import OmegaConf
from itertools import islice
from ldm.util import instantiate_from_config
import torch.nn as nn


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=True):
    pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def get_models(args):
    from pathlib import Path

    config = Path(args.config)
    ckpt = Path(args.ckpt)

    print(f"Loading model from {ckpt}")

    sd = load_model_from_config(ckpt)
    lo, li = [], []
    for key, _ in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)

    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")
    config.modelUNet.params.ddim_steps = args.ddim_steps
    config.modelUNet.params.small_batch = args.small_batch

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()

    cs_model = instantiate_from_config(config.modelCondStage)
    _, _ = cs_model.load_state_dict(sd, strict=False)
    cs_model.eval()

    fs_model = instantiate_from_config(config.modelFirstStage)
    _, _ = fs_model.load_state_dict(sd, strict=False)
    fs_model.eval()

    if args.half_precision:
        model = model.half()
        cs_model = cs_model.half()
        fs_model = fs_model.half()

    return model, cs_model, fs_model


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale
