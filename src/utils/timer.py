import contextlib
import math
import os
import time

import psutil


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
            print(f"[{name}] done [{m1:.1f}GB({p1:.1f}%)({sign}{delta:.3f}GB)] {t1 - t0:.4f} s")

    @staticmethod
    def get_metric() -> tuple[float, float, float]:
        t = time.time()
        p = psutil.Process(os.getpid())
        m: float = p.memory_info()[0] / 2.0**30
        per: float = psutil.virtual_memory().percent
        return t, m, per
