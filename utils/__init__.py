from .data import *
from .notify import *


import os
import random

import numpy as np
import torch
import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    """
    Examples:
        >>> with timer("wait"):
                time.sleep(2.0)
    """
    t0 = time.time()
    yield
    elapsed_time = time.time() - t0
    print(f"[{name}] done in {elapsed_time:.1f} s")


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True