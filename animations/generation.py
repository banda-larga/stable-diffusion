import os, sys
import numpy as np
import time

import torch
from contextlib import nullcontext
from einops import rearrange, repeat
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import sys

import sampling
from compvis import CompVisDenoiser
from models import CFGDenoiser
from utils import load_img, make_callback


def generate(args, models, return_latent=False, return_sample=False, return_c=False):
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    model, cs_model, fs_model = models

    if args.sampler == "plms":
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = CompVisDenoiser(model)

    batch_size = args.n_samples
    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    init_latent = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        if args.less_vram:
            fs_model.to(args.device)
            init_latent = fs_model.get_first_stage_encoding(
                fs_model.encode_first_stage(args.init_sample)
            )
            mem = torch.cuda.memory_allocated() / 1e6
            print(f"fs_model to cpu: {mem} MB")
            fs_model.to("cpu")
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(4)
        else:
            init_latent = fs_model.get_first_stage_encoding(
                fs_model.encode_first_stage(args.init_sample)
            )

    elif args.init_image != None and args.init_image != "":
        init_image = load_img(args.init_image, shape=(args.W, args.H)).to(args.device)
        init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
        if args.less_vram:
            fs_model.to(args.device)
            init_latent = fs_model.get_first_stage_encoding(
                fs_model.encode_first_stage(init_image)
            )
            mem = torch.cuda.memory_allocated() / 1e6
            print(f"fs_model to cpu: {mem} MB")
            fs_model.to("cpu")
            while torch.cuda.memory_allocated() / 1e6 >= mem:
                time.sleep(4)
        else:
            init_latent = fs_model.get_first_stage_encoding(
                fs_model.encode_first_stage(init_image)
            )

    sampler.make_schedule(
        ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, verbose=False
    )

    t_enc = int((1.0 - args.strength) * args.steps)

    start_code = None
    if args.fixed_code and init_latent == None:
        start_code = torch.randn(
            [args.n_samples, args.C, args.H // args.f, args.W // args.f],
            device=args.device,
        )

    callback = make_callback(
        sampler=args.sampler,
        dynamic_threshold=args.dynamic_threshold,
        static_threshold=args.static_threshold,
    )

    results = []
    precision_scope = autocast if args.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            for n in range(args.n_samples):
                for prompts in data:
                    uc = None
                    if args.less_vram:
                        cs_model.to(args.device)

                    if args.scale != 1.0:
                        uc = cs_model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = cs_model.get_learned_conditioning(prompts)

                    if args.init_c != None:
                        c = args.init_c

                    if args.sampler in [
                        "klms",
                        "dpm2",
                        "dpm2_ancestral",
                        "heun",
                        "euler",
                        "euler_ancestral",
                    ]:
                        shape = [args.C, args.H // args.f, args.W // args.f]
                        sigmas = model_wrap.get_sigmas(args.steps)
                        if args.use_init:
                            sigmas = sigmas[len(sigmas) - t_enc - 1 :]
                            x = (
                                init_latent
                                + torch.randn(
                                    [args.n_samples, *shape], device=args.device
                                )
                                * sigmas[0]
                            )
                        else:
                            x = (
                                torch.randn(
                                    [args.n_samples, *shape], device=args.device
                                )
                                * sigmas[0]
                            )
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {
                            "cond": c,
                            "uncond": uc,
                            "cond_scale": args.scale,
                        }
                        if args.sampler == "klms":
                            samples = sampling.sample_lms(
                                model_wrap_cfg,
                                x,
                                sigmas,
                                extra_args=extra_args,
                                disable=False,
                                callback=callback,
                            )
                        elif args.sampler == "dpm2":
                            samples = sampling.sample_dpm_2(
                                model_wrap_cfg,
                                x,
                                sigmas,
                                extra_args=extra_args,
                                disable=False,
                                callback=callback,
                            )
                        elif args.sampler == "dpm2_ancestral":
                            samples = sampling.sample_dpm_2_ancestral(
                                model_wrap_cfg,
                                x,
                                sigmas,
                                extra_args=extra_args,
                                disable=False,
                                callback=callback,
                            )
                        elif args.sampler == "heun":
                            samples = sampling.sample_heun(
                                model_wrap_cfg,
                                x,
                                sigmas,
                                extra_args=extra_args,
                                disable=False,
                                callback=callback,
                            )
                        elif args.sampler == "euler":
                            samples = sampling.sample_euler(
                                model_wrap_cfg,
                                x,
                                sigmas,
                                extra_args=extra_args,
                                disable=False,
                                callback=callback,
                            )
                        elif args.sampler == "euler_ancestral":
                            samples = sampling.sample_euler_ancestral(
                                model_wrap_cfg,
                                x,
                                sigmas,
                                extra_args=extra_args,
                                disable=False,
                                callback=callback,
                            )

                    else:
                        if init_latent != None:
                            z_enc = sampler.stochastic_encode(
                                init_latent,
                                torch.tensor([t_enc] * batch_size).to(args.device),
                            )
                            samples = sampler.decode(
                                z_enc,
                                c,
                                t_enc,
                                unconditional_guidance_scale=args.scale,
                                unconditional_conditioning=uc,
                            )
                        else:
                            if args.sampler == "plms" or args.sampler == "ddim":
                                shape = [args.C, args.H // args.f, args.W // args.f]
                                samples, _ = sampler.sample(
                                    S=args.steps,
                                    conditioning=c,
                                    batch_size=args.n_samples,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=args.scale,
                                    unconditional_conditioning=uc,
                                    eta=args.ddim_eta,
                                    x_T=start_code,
                                    img_callback=callback,
                                )
                    if args.less_vram:
                        mem = torch.cuda.memory_allocated() / 1e6
                        cs_model.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(2)
                        print("cs_model to cpu")

                    if return_latent:
                        results.append(samples.clone())
                    if args.less_vram:
                        fs_model.to(args.device)

                    x_samples = fs_model.decode_first_stage(samples)

                    if args.less_vram:
                        mem = torch.cuda.memory_allocated() / 1e6
                        fs_model.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(2)
                        print("cs_model to cpu")

                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        x_sample = 255.0 * rearrange(
                            x_sample.cpu().numpy(), "c h w -> h w c"
                        )
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        results.append(image)
    return results
