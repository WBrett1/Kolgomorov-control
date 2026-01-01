# -----------------------------------------------------------------------------
# This file is based on the original implementation from:
# https://pypi.org/project/kolsol/
#
# It has been adapted and modified for use in the current project.
# -----------------------------------------------------------------------------

from typing import TypeVar

import numpy as np
import torch


TypeTensor = TypeVar('TypeTensor', np.ndarray, torch.Tensor)