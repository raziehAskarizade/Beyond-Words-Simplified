# Fardin Rastakhiz @ 2023

from torch import tensor
from typing import Dict


class HeteroLossArgs:
    def __init__(self, y:tensor, x_dict:Dict[str, tensor]):
        self.y = y
        self.x_dict = x_dict