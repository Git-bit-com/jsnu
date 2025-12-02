from __future__ import annotations

import os
import json
import time
import random
import logging
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd

def _get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    return _get_logger(__name__, level)
logger = setup_logger()

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def ensure_dir(path: str | None) -> str:
    dirpath = path or "."
    os.makedirs(dirpath, exist_ok=True)
    return dirpath


@contextmanager
def time_block(task: str = ""):
    t0 = time.time()
    if task:
        logger.info(f"⏳ {task} 开始...")
    try:
        yield
    finally:
        dt = time.time() - t0
        if task:
            logger.info(f"✅ {task} 完成，用时 {dt:.2f}s")
        else:
            logger.info(f"✅ 完成，用时 {dt:.2f}s")

def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件未找到: {path}")

    kwargs.setdefault("low_memory", False)

    encodings = ["utf-8", "utf-8-sig", "gbk", "cp936"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        msg = (
            f"无法用常见编码读取文件 {path}。最后错误：{last_err}. "
            f"请手动指定 encoding=... 或检查文件编码。原始异常：{e}"
        )
        raise UnicodeDecodeError("safe_read_csv", b"", 0, 1, msg)


def safe_write_csv(
    df: pd.DataFrame,
    path: str,
    index: bool = False,
    encoding: str = "utf-8",
    na_rep: str = "",
    **kwargs,
) -> str:
    dirpath = os.path.dirname(path)
    ensure_dir(dirpath)
    tmp_path = f"{path}.tmp"
    kwargs.setdefault("float_format", None)
    df.to_csv(tmp_path, index=index, encoding=encoding, na_rep=na_rep, **kwargs)
    os.replace(tmp_path, path)
    logger.info(f"CSV 已保存：{path}（{len(df)} 行）")
    return path


def save_submission(df: pd.DataFrame, path: str = "output/submission.csv") -> str:
    return safe_write_csv(df, path, index=False, encoding="utf-8", na_rep="")

def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str, ensure_ascii: bool = False, indent: int = 2) -> str:
    dirpath = os.path.dirname(path)
    ensure_dir(dirpath)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
    os.replace(tmp_path, path)
    logger.info(f"💾 JSON 已保存：{path}")
    return path

def to_int_safe(x):
    try:
        return int(float(x))
    except Exception:
        return x
