"""
filter_4_2.parquetなどのデータを読み込んで、embeddingを計算し、faissのindexを作成する。
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm.auto import tqdm

import re
import faiss
from faiss import read_index, write_index

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import ctypes
import os

libc = ctypes.CDLL("libc.so.6")
sys.path.append(os.pardir)

import utils


# hydraで設定を読み込む
@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c: DictConfig) -> None:
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.preprocess

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.preprocess.split('/')[-1]}"
    preprocessed_path = Path(f"./preprocessed/{exp_name}")
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    df = pd.read_parquet(cfg.filter_wiki_path)
    if cfg.debug:
        df = df.head(10)

    # モデル読み込み
    model = SentenceTransformer(cfg.sim_model, device="cuda")
    model.max_seq_length = cfg.max_length
    model.half()

    # embedding計算
    df["concat_text"] = df["title"].fillna("") + " > " + df["section"].fillna("") + " > " + df["text"].fillna("")
    embeddings = model.encode(
        df["concat_text"].tolist(),
        batch_size=cfg.batch_size,
        device="cuda",
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )
    np.save(str(preprocessed_path / "embeddings.npy"), embeddings)
    """
    embeddings = np.load(str(preprocessed_path / "embeddings.npy"))
    """

    # 型をfloat32に変換
    embeddings = embeddings.astype(np.float32)
    print("embeddings.shape:", embeddings.shape)

    # index作成
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, cfg.nlist, cfg.n_subquantizer, cfg.n_bits)

    with utils.timer("faiss train"):
        """
        res = faiss.StandardGpuResources()  # use a single GPU
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
        """

        assert not index.is_trained
        index.train(
            embeddings,
        )
        assert index.is_trained

    with utils.timer("faiss add"):
        index.add(embeddings)

    with utils.timer("faiss write"):
        # to cpu
        index = faiss.index_gpu_to_cpu(index)
        write_index(index, str(preprocessed_path / "ivfpq_index.faiss"))


if __name__ == "__main__":
    main()
