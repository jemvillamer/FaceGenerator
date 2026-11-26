"""Shared UI constants."""

THUMB_VISIBLE = 6          # thumbnail slots shown before the "+N" tile
THUMB_SIZE = 68
MAX_IMAGES = 1000

# Gender values the engine understands, in button order (button id = index).
# The engine gets these values, not the button labels.
GENDERS = ["Any", "Female", "Male"]

# Filter defaults: applied at startup and by Reset, so the two always match.
DEFAULT_AGE = "Any"        # an age bucket key (see latents.AGE_BUCKETS) or "Any"
DEFAULT_GENDER = "Any"     # one of GENDERS
DEFAULT_IMAGE_COUNT = 20
