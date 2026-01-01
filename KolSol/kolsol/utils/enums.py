# -----------------------------------------------------------------------------
# This file is based on the original implementation from:
# https://pypi.org/project/kolsol/
#
# It has been adapted and modified for use in the current project.
# -----------------------------------------------------------------------------

import enum


class eDirection(enum.IntEnum):
    i: int = 0
    j: int = 1
    k: int = 2