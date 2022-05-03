"""
Updated soot foil image processing module

This version includes exclusion zones to allow for more measurements to be
collected even on sketchier foils with clear "bad measurement" zones. It
also has the API separated from the actual calculation functions, which is
something I wish I'd known enough to do earlier. This new version of the module
isn't being retrofitted into the main funcs module yet because frankly I don't
feel like it.
"""
from .api import collect_shot_deltas, Shot

__all__ = ["collect_shot_deltas", "Shot"]
