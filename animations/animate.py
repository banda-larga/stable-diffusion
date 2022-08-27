import json
from IPython import display

import os, pathlib, subprocess, time
import cv2
import numpy as np
import pandas as pd
import random
from dataclasses import asdict
import gc

from utils import (
    make_xform_2d,
    get_inbetweens,
    parse_key_frames,
    sample_to_cv2,
    maintain_colors,
    sample_from_cv2,
    add_noise,
)
from models import get_models
from generation import generate


def next_seed(args):
    if args.seed_behavior == "iter":
        args.seed += 1
    elif args.seed_behavior == "fixed":
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, 2**32)
    return args.seed


def animate(args, anim_args):

    args.timestring = time.strftime("%Y%m%d%H%M%S")
    args.strength = max(0.0, min(1.0, args.strength))

    if args.seed == -1:
        args.seed = random.randint(0, 2**32)
    if anim_args.animation_mode == "Video Input":
        args.use_init = True
    if not args.use_init:
        args.init_image = None
        args.strength = 0
    if args.sampler == "plms" and (args.use_init or anim_args.animation_mode != "None"):
        print(f"Init images aren't supported with PLMS yet, switching to KLMS")
        args.sampler = "klms"
    if args.sampler != "ddim":
        args.ddim_eta = 0

    if anim_args.animation_mode == "None":
        anim_args.max_frames = 1

    if anim_args.key_frames:
        anim_args.angle_series = get_inbetweens(args, parse_key_frames(anim_args.angle))
        anim_args.zoom_series = get_inbetweens(args, parse_key_frames(anim_args.zoom))
        anim_args.translation_x_series = get_inbetweens(
            args, parse_key_frames(anim_args.translation_x)
        )
        anim_args.translation_y_series = get_inbetweens(
            args, parse_key_frames(anim_args.translation_y)
        )

    if anim_args.animation_mode == "2D":
        render_animation(args, anim_args)
    elif anim_args.animation_mode == "Video Input":
        render_input_video(args, anim_args)
    elif anim_args.animation_mode == "Interpolation":
        render_interpolation(args, anim_args)
    else:
        render_image_batch(args)


def render_image_batch(args):
    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_settings or args.save_samples:
        print(f"Saving to {os.path.join(args.outdir, args.timestring)}_*")

    # save settings for the batch
    if args.save_settings:
        filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(dict(args.__dict__), f, ensure_ascii=False, indent=4)

    index = 0

    # function for init image batching
    init_array = []
    if args.use_init:
        if args.init_image == "":
            raise FileNotFoundError("No path was given for init_image")
        if args.init_image.startswith("http://") or args.init_image.startswith(
            "https://"
        ):
            init_array.append(args.init_image)
        elif not os.path.isfile(args.init_image):
            if (
                args.init_image[-1] != "/"
            ):  # avoids path error by adding / to end if not there
                args.init_image += "/"
            for image in sorted(
                os.listdir(args.init_image)
            ):  # iterates dir and appends images to init_array
                if image.split(".")[-1] in ("png", "jpg", "jpeg"):
                    init_array.append(args.init_image + image)
        else:
            init_array.append(args.init_image)
    else:
        init_array = [""]

    for batch_index in range(args.n_batch):
        print(f"Batch {batch_index+1} of {args.n_batch}")

        for image in init_array:  # iterates the init images
            args.init_image = image
            for prompt in args.prompts:
                args.prompt = prompt
                results = generate(args)
                for image in results:
                    if args.save_samples:
                        filename = f"{args.timestring}_{index:05}_{args.seed}.png"
                        image.save(os.path.join(args.outdir, filename))
                    if args.display_samples:
                        display.display(image)
                    index += 1
                args.seed = next_seed(args)


def render_animation(args, anim_args):
    models = get_models(args)

    if not args.less_vram:
        models = (model.to(args.device) for model in models)

    # animations use key framed prompts
    args.prompts = args.animation_prompts

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to {args.outdir}")

    # save settings for the batch
    settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
    with open(settings_filename, "w+", encoding="utf-8") as f:
        s = {**asdict(args), **asdict(anim_args)}
        json.dump(s, f, ensure_ascii=True, indent=4)

    # expand prompts out to per-frame
    prompt_series = pd.Series([np.nan for _ in range(args.max_frames)])
    for i, prompt in args.animation_prompts:
        prompt_series[i] = prompt
    prompt_series = prompt_series.ffill().bfill()

    # check for video inits
    using_vid_init = anim_args.animation_mode == "Video Input"

    args.n_samples = 1
    prev_sample = None
    color_match_sample = None
    for frame_idx in range(args.max_frames):
        print(f"Rendering animation frame {frame_idx} of {args.max_frames}")

        # apply transforms to previous frame
        if prev_sample is not None:
            if anim_args.key_frames:
                angle = anim_args.angle_series[frame_idx]
                zoom = anim_args.zoom_series[frame_idx]
                translation_x = anim_args.translation_x_series[frame_idx]
                translation_y = anim_args.translation_y_series[frame_idx]
                print(
                    f"angle: {angle}",
                    f"zoom: {zoom}",
                    f"translation_x: {translation_x}",
                    f"translation_y: {translation_y}",
                )
            xform = make_xform_2d(
                args.W, args.H, translation_x, translation_y, angle, zoom
            )

            # transform previous frame
            prev_img = sample_to_cv2(prev_sample)
            prev_img = cv2.warpPerspective(
                prev_img,
                xform,
                (prev_img.shape[1], prev_img.shape[0]),
                borderMode=cv2.BORDER_WRAP
                if anim_args.border == "wrap"
                else cv2.BORDER_REPLICATE,
            )

            # apply color matching
            if anim_args.color_coherence == "MatchFrame0":
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(
                        prev_img, color_match_sample, (frame_idx % 2) == 0
                    )

            # apply frame noising
            noised_sample = add_noise(
                sample_from_cv2(prev_img), anim_args.previous_frame_noise
            )

            # use transformed previous frame as init for current
            args.use_init = True
            args.init_sample = noised_sample.half().to(args.device)
            args.strength = max(0.0, min(1.0, anim_args.previous_frame_strength))

        # grab prompt for current frame
        args.prompt = prompt_series[frame_idx]
        print(f"{args.prompt} {args.seed}")

        # grab init image for current frame
        if using_vid_init:
            init_frame = os.path.join(
                args.outdir, "inputframes", f"{frame_idx+1:04}.jpg"
            )
            print(f"Using video init frame {init_frame}")
            args.init_image = init_frame

        # sample the diffusion model
        results = generate(args, models=models, return_latent=False, return_sample=True)
        sample, image = results[0], results[1]

        filename = f"{args.timestring}_{frame_idx:05}.png"
        print(f"Saving frame {filename}")
        image.save(os.path.join(args.outdir, filename))
        print(f"Saved frame {filename}")
        if not using_vid_init:
            prev_sample = sample

        args.seed = next_seed(args)
        gc.collect()


def render_input_video(args, anim_args):
    # create a folder for the video input frames to live in
    video_in_frame_path = os.path.join(args.outdir, "inputframes")
    os.makedirs(os.path.join(args.outdir, video_in_frame_path), exist_ok=True)

    # save the video frames from input video
    print(
        f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {video_in_frame_path}..."
    )
    try:
        for f in pathlib.Path(video_in_frame_path).glob("*.jpg"):
            f.unlink()
    except:
        pass
    vf = f"select=not(mod(n\,{anim_args.extract_nth_frame}))"
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"{anim_args.video_init_path}",
            "-vf",
            f"{vf}",
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            "-loglevel",
            "error",
            "-stats",
            os.path.join(video_in_frame_path, "%04d.jpg"),
        ],
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")

    # determine max frames from length of input frames
    anim_args.max_frames = len(
        [f for f in pathlib.Path(video_in_frame_path).glob("*.jpg")]
    )

    args.use_init = True
    print(
        f"Loading {anim_args.max_frames} input frames from {video_in_frame_path} and saving video frames to {args.outdir}"
    )
    render_animation(args, anim_args)


def render_interpolation(args, anim_args):
    models = get_models(args)

    if not args.less_vram:
        models = (model.to(args.device) for model in models)

    # animations use key framed prompts
    animation_prompts = args.animation_prompts
    assert args.animation_prompts is not None

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to {args.outdir}")

    # save settings for the batch
    settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
    with open(settings_filename, "w+", encoding="utf-8") as f:
        s = {**dict(args.__dict__), **dict(anim_args.__dict__)}
        json.dump(s, f, ensure_ascii=False, indent=4)

    # Interpolation Settings
    args.n_samples = 1
    args.seed_behavior = (
        "fixed"  # force fix seed at the moment bc only 1 seed is available
    )
    prompts_c_s = []  # cache all the text embeddings

    print(f"Preparing for interpolation of the following...")

    for i, prompt in animation_prompts.items():
        args.prompt = prompt

        # sample the diffusion model
        results = generate(args, models=models, return_c=True)
        c, image = results[0], results[1]
        prompts_c_s.append(c)

        # display.clear_output(wait=True)
        display.display(image)

        args.seed = next_seed(args)

    display.clear_output(wait=True)
    print(f"Interpolation start...")

    frame_idx = 0

    for i in range(len(prompts_c_s) - 1):
        for j in range(anim_args.interpolate_x_frames + 1):
            # interpolate the text embedding
            prompt1_c = prompts_c_s[i]
            prompt2_c = prompts_c_s[i + 1]
            args.init_c = prompt1_c.add(
                prompt2_c.sub(prompt1_c).mul(
                    j * 1 / (anim_args.interpolate_x_frames + 1)
                )
            )

            results = generate(args, models=models)
            image = results[0]

            filename = f"{args.timestring}_{frame_idx:05}.png"
            image.save(os.path.join(args.outdir, filename))
            frame_idx += 1

            display.clear_output(wait=True)
            display.display(image)

            args.seed = next_seed(args)

    # generate the last prompt
    args.init_c = prompts_c_s[-1]
    results = generate(args, models=models)
    image = results[0]
    filename = f"{args.timestring}_{frame_idx:05}.png"
    image.save(os.path.join(args.outdir, filename))

    display.clear_output(wait=True)
    display.display(image)
    args.seed = next_seed(args)

    # clear init_c
    args.init_c = None
