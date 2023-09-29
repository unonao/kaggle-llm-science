import ctypes
import gc
import os
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from datasets import load_dataset, load_from_disk
import blingfire as bf
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


configs = [
    {"window_size": 10, "sliding_size": 10},  # ほぼそのまま入力
    {"window_size": 4, "sliding_size": 2},
    {"window_size": 2, "sliding_size": 1},
]


def extract_chunk_by_sliding_window(text: str, window_size: int, sliding_size: int) -> list[str]:
    """
    text のリストをsliding windowで結合する。window_size個のtextが含まれるまで結合し、sliding_size個ずつずらして結合する。
    """
    _, sentence_offsets = bf.text_to_sentences_and_offsets(text)
    text_list = []
    for o in sentence_offsets:
        if 3 < o[1] - o[0]:
            text_list.append(text[o[0] : o[1]])

    chunks = []
    for i in range(0, len(text_list), sliding_size):
        chunk = " ".join(text_list[i : i + window_size])
        chunks.append(chunk)
    return chunks


dir_path = Path(f"preprocessed/{Path(sys.argv[0]).stem}")
dir_path.mkdir(exist_ok=True, parents=True)
paraphs_parsed_dataset = load_from_disk("input/all-paraphs-parsed-expanded")


for config in configs:
    print(config)
    df = pd.DataFrame(paraphs_parsed_dataset)
    df["text"] = df["text"].parallel_apply(
        lambda x: extract_chunk_by_sliding_window(x, config["window_size"], config["sliding_size"])
    )
    explode_df = df.explode(["text"]).reset_index(drop=True)

    # 保存
    explode_df.to_parquet(dir_path / f"filter_{config['window_size']}_{config['sliding_size']}.parquet")
