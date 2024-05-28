import copy
import functools
import logging
import math
import os
import pickle
import random
import signal
import sys
import time
from contextlib import contextmanager
from typing import Union, List

import numpy as np
import torch
from tensorboardX import SummaryWriter

def save_pickle(obj, path: str, open_options: str = "wb"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, open_options) as f:
        pickle.dump(obj, f)
    f.close()