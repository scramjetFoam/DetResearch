from typing import Optional
import numpy as np

# noinspection PyUnresolvedReferences
from .get_px_deltas_from_lines import get_px_deltas_from_lines as rgpfl


def fast_get_px_deltas_from_lines(
    img_path: str,
    mask_path: Optional[str] = None
) -> np.array:
    return rgpfl(img_path, mask_path)
