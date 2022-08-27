from dataclasses import dataclass, field
from typing import Dict, List
from PIL import Image
from utils import get_output_folder
import torch


@dataclass
class Settings:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config: str = "../v1-inference.yaml"
    ckpt: str = "../../models/ldm/stable-diffusion-v1/model.ckpt"

    animation_prompts: List = field(
        default_factory=lambda: [
            "hello world",
            "a painting of a virus monster playing guitar",
        ]
    )
    prompt_ratios: List = field(default_factory=lambda: [0.3, 0.7])

    seeds: List = field(
        default_factory=lambda: [
            42,
            42,
        ]
    )
    n_interpolate: int = 1
    max_frames = 1000
    strength: float = 0.5

    input_image: Image = None
    use_init = False

    # generation params
    seed: int = 13
    fixed_code: bool = True
    ddim_steps: int = 50
    plms: bool = False
    ddim_eta: float = 0.0
    C: int = 4
    f: int = 8
    scale: float = 12.5
    dyn: float = None

    H: int = 256
    W: int = 256
    W, H = map(lambda x: x - x % 64, (W, H))

    n_iter: int = 1
    n_samples: int = 1
    n_batch: int = 1

    batch_name = "Animations"
    output_path = "./output"
    outdir = get_output_folder(output_path, batch_name)

    small_batch = False

    dynamic_threshold = None
    static_threshold = None

    init_latent = None
    init_sample = None
    init_c = None

    save_grid = False
    save_samples = True
    save_settings = True

    sampler = "klms"
    seed_behavior = "iter"
    precision = "autocast"
    fixed_code = True
    prompt = ""
    timestring = ""
    steps: int = 100
    less_vram: bool = True

    interp_spline = "Linear"
    half_precision = True


@dataclass
class AnimationSettings:

    animation_mode = "2D"
    border = "wrap"
    key_frames = True

    # Do not change, currently will not look good.

    angle: str = "0:(0)"
    zoom: str = "0: (1.04)"
    translation_x: str = "0: (0)"
    translation_y: str = "0: (0)"

    # Coherence
    color_coherence = "MatchFrame0"
    previous_frame_noise = 0.02
    previous_frame_strength = 0.65

    # Video Input:
    video_init_path = "/content/video_in.mp4"
    extract_nth_frame = 1

    # Interpolation:
    interpolate_x_frames = 4
