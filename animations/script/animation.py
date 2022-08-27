import sys

sys.path.append("..")

from settings import Settings, AnimationSettings
from animate import animate

animation_prompts = [
    (
        0,
        "Portrait of a woman with a beautiful dress, painting by Rogier van der Weyden, trending on artstation HQ",
    ),
    (
        10,
        "Portrait of a man wearing an old dress, painting by Rogier van der Weyden, trending on artstation HQ",
    ),
]

args = Settings(
    n_interpolate=1,
    fixed_code=True,
    steps=50,
    W=640,
    H=640,
    animation_prompts=animation_prompts,
)

anim_args = AnimationSettings(
    angle="0:(0)",
    zoom="0: (1.04)",
    translation_x="0: (2.5)",
    translation_y="0: (0)",
)

animate(args, anim_args)
