import os, time
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from skimage.exposure import match_histograms
from einops import rearrange


def make_xform_2d(width, height, translation_x, translation_y, angle, scale):
    center = (width // 2, height // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return np.matmul(rot_mat, trans_mat)


def parse_key_frames(string, prompt_parser=None):
    import re

    pattern = r"((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])"
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()["frame"])
        param = match_object.groupdict()["param"]
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError("Key Frame string not correctly formatted")
    return frames


def get_inbetweens(args, key_frames, integer=False):
    key_frame_series = pd.Series([np.nan for a in range(args.max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    interp_method = args.interp_spline
    if interp_method == "Cubic" and len(key_frames.items()) <= 3:
        interp_method = "Quadratic"
    if interp_method == "Quadratic" and len(key_frames.items()) <= 2:
        interp_method = "Linear"

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[args.max_frames - 1] = key_frame_series[
        key_frame_series.last_valid_index()
    ]

    key_frame_series = key_frame_series.interpolate(
        method=interp_method.lower(), limit_direction="both"
    )

    if integer:
        return key_frame_series.astype(int)
    return key_frame_series


def add_noise(sample: torch.Tensor, noise_amt: float):
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt


def get_output_folder(output_path, batch_folder=None):
    yearMonth = time.strftime("%Y-%m/")
    out_path = os.path.join(output_path, yearMonth)
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
        # we will also make sure the path suffix is a slash if linux and a backslash if windows
        if out_path[-1] != os.path.sep:
            out_path += os.path.sep
    os.makedirs(out_path, exist_ok=True)
    return out_path


def load_img(path, shape):
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(path).convert("RGB")

    image = image.resize(shape, resample=Image.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def maintain_colors(prev_img, color_match_sample, hsv=False):
    if hsv:
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else:
        return match_histograms(prev_img, color_match_sample, multichannel=True)


def make_callback(sampler, dynamic_threshold=None, static_threshold=None):
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image after each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1, img.ndim)))
        s = np.max(np.append(s, 1.0))
        torch.clamp_(img, -1 * s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback(args_dict):
        if static_threshold is not None:
            torch.clamp_(args_dict["x"], -1 * static_threshold, static_threshold)
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict["x"], dynamic_threshold)

    # Function that is called on the image (img) and step (i) at each step
    def img_callback(img, i):
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1 * static_threshold, static_threshold)

    if sampler in ["plms", "ddim"]:
        # Callback function formated for compvis latent diffusion samplers
        callback = img_callback
    else:
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback

    return callback


def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample


def sample_to_cv2(sample: torch.Tensor) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(
        np.float32
    )
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255).astype(np.uint8)
    return sample_int8
