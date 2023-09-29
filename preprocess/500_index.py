"""
指定したデータを読み取り pandas の DataFrame に変換して保存。
各行のembeddingを計算し、faissのindexを作成する。
"""

import ctypes
import gc
import os
import re
import sys
import time

from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

import faiss
from faiss import read_index, write_index
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from datasets import load_dataset, load_from_disk
from omegaconf import DictConfig, OmegaConf


# hydraで設定を読み込む
@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c: DictConfig) -> None:
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.preprocess

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.preprocess.split('/')[-1]}"
    preprocessed_path = Path(f"./preprocessed/{exp_name}")
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    print(cfg)
    print("preprocessed_path:", preprocessed_path)

    df_path = preprocessed_path.parent / f"{Path(cfg.dataset_path).stem}.parquet"

    if df_path.exists():
        df = pd.read_parquet(df_path)
    else:
        # データセットの読み込み
        dataset = load_from_disk(cfg.dataset_path)
        # データセットの前処理
        df = pd.DataFrame(dataset)
        if "section" not in df.columns:
            df["section"] = ""
        df["context"] = df["title"].fillna("") + " > " + df["section"].fillna("") + " > " + df["text"].fillna("")
        df["context"] = df["context"].str.replace("\n", " ")
        df = df.drop(["title", "section", "text"], axis=1)
        # データセットの保存(preprocessed_pathの親)
        df.to_parquet(df_path)
    if cfg.debug:
        df = df.head(10)

    # モデル読み込み
    model = SentenceTransformer(cfg.sim_model, device="cuda")
    model.max_seq_length = cfg.max_length
    model.half()

    # embedding計算
    embeddings = model.encode(
        df["context"].tolist(),
        batch_size=cfg.batch_size,
        device="cuda",
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype(np.float32)
    print("embeddings.shape:", embeddings.shape)

    # index作成
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print("index.is_trained:", index.is_trained)
    print("index.ntotal:", index.ntotal)

    # index保存
    faiss.write_index(index, str(preprocessed_path / "index.faiss"))
    print("index saved")


if __name__ == "__main__":
    main()
