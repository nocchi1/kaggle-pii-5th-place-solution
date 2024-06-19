import os
import sys
import psutil
import time
import random
import math
import numpy as np
import contextlib
from pathlib import Path, PosixPath
from typing import Tuple
import torch
from omegaconf import OmegaConf


class TimeUtil:
    @staticmethod
    @contextlib.contextmanager
    def timer(name: str, logger=None):
        t0, m0, p0 = TimeUtil.get_metric()
        if logger is not None:
            logger.info(f"[{name}] start [{m0:.1f}GB({p0:.1f}%)]")
        else:
            print(f"[{name}] start [{m0:.1f}GB({p0:.1f}%)]")
        yield
        t1, m1, p1 = TimeUtil.get_metric()
        delta = m1 - m0
        sign = "+" if delta >= 0 else "-"
        delta = math.fabs(delta)
        if logger is not None:
            logger.info(
                f"[{name}] done [{m1:.1f}GB({p1:.1f}%)({sign}{delta:.3f}GB)] {t1 - t0:.4f} s"
            )
        else:
            print(
                f"[{name}] done [{m1:.1f}GB({p1:.1f}%)({sign}{delta:.3f}GB)] {t1 - t0:.4f} s"
            )

    @staticmethod
    def get_metric() -> Tuple[float, float, float]:
        t = time.time()
        p = psutil.Process(os.getpid())
        m: float = p.memory_info()[0] / 2.0**30
        per: float = psutil.virtual_memory().percent
        return t, m, per


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def get_logger(log_dir: Path, stdout: bool=True):
    from loguru import logger
    logger.remove() # デフォルトの設定を削除
    custom_format = "[ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level} ] {message}</level>"
    logger.add(log_dir / 'log_{time:YYYY-MM-DD-HH-mm-ss}.txt', level="INFO", colorize=False, format=custom_format)
    if stdout:
        custom_format = "[ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level} ] {message}</level>"
        logger.add(sys.stdout, level="INFO", colorize=True, format=custom_format)
    return logger