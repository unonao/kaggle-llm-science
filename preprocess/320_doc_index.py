# embeddingとparquetをまとめて、faiss indexを作成する。メモリが多く必要

import gc
import os
import sys
from pathlib import Path
import glob

import faiss
import hydra
import numpy as np
import pandas as pd
from faiss import read_index, write_index
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

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
    # フォルダ作成
    os.makedirs(preprocessed_path, exist_ok=True)

    print(cfg)
    print("preprocessed_path:", preprocessed_path)

    # numpy, parquet データの読み込み・結合・float32化
    npy_paths = [path for path in glob.glob(f"{cfg.embedding_dir}/*.npy") if "all" not in path]
    npy_paths.sort()

    all_embeddings_list = []
    all_df_list = []

    for path in tqdm(npy_paths):
        if cfg.debug:
            if len(all_embeddings_list) > 1:
                break

        print(path)
        # parquet の拡張子にして読み込み
        parquet_path = path.replace(".npy", ".parquet")
        df = pd.read_parquet(parquet_path).reset_index(drop=True)

        # numpy を読み込み
        embeddings = np.load(path)
        embeddings = embeddings.astype(np.float32, copy=False)

        # 結合
        all_embeddings_list.append(embeddings)
        df["file"] = parquet_path.split("/")[-1]
        all_df_list.append(df[["id", "file"]])
        del embeddings
        del df
        gc.collect()

    # 結合したデータを保存
    print("concatenate embeddings")
    all_embeddings = np.concatenate(
        all_embeddings_list,
        axis=0,
    )
    del all_embeddings_list
    gc.collect()
    print("all_embeddings :", all_embeddings.nbytes / (1024**3), " GB")
    all_embeddings = all_embeddings.astype(np.float32, copy=False)
    print("all_embeddings :", all_embeddings.nbytes / (1024**3), " GB")
    gc.collect()
    np.save(preprocessed_path / "all.npy", all_embeddings)

    print("concatenate df")
    all_df = pd.concat(all_df_list, axis=0)
    all_df.reset_index(drop=True, inplace=True)
    all_df.to_parquet(preprocessed_path / "all.parquet")
    gc.collect()

    # faiss index の作成
    print("faiss index")
    dim = all_embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, cfg.nlist, cfg.n_subquantizer, cfg.n_bits)

    with utils.timer("faiss train"):
        res = faiss.StandardGpuResources()  # use a single GPU
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)

        assert not index.is_trained
        index.train(
            all_embeddings,
        )
        assert index.is_trained

    with utils.timer("faiss add"):
        index.add(all_embeddings)

    with utils.timer("faiss write"):
        # to cpu
        index = faiss.index_gpu_to_cpu(index)
        write_index(index, str(preprocessed_path / "ivfpq_index.faiss"))


if __name__ == "__main__":
    main()
