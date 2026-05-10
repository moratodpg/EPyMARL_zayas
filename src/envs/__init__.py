import os
import sys

from .multiagentenv import MultiAgentEnv
from .structzayas_wrapper import StructZayasWrapper

def structzayas_fn(**kwargs) -> MultiAgentEnv:
        return StructZayasWrapper(**kwargs)


REGISTRY = {}
REGISTRY["struct_zayas"] = structzayas_fn