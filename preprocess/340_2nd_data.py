"""
1st の学習結果を使って 2nd のデータを作成する

1. 元データと1stの結果読み込み
2. tf-idf なども用いて特徴量を作成。集約特徴量も作成
3. (prompt_id, option) で一意の行になるように分割
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm.auto import tqdm

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_tfidf(row: dict[str, str]) -> np.ndarray:
    """
    tfidfを計算する
    """
    # tfidfの計算
    tfidf = TfidfVectorizer()
    base_cols = ["A", "B", "C", "D", "E"]
    fit_cols = base_cols + ["context", "prompt"]
    tfidf_vec = tfidf.fit([row[col] for col in fit_cols])
    # base_cols と context の 類似度を計算
    base_vec = tfidf_vec.transform([row[col] for col in base_cols])
    context_vec = tfidf_vec.transform([row["context"]])
    sim = cosine_similarity(base_vec, context_vec)
    return sim


def make_dataset(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """
    元データと1stの予測結果を結合して2ndのデータを作成する
    """
    # まずはtfidf
    print("tfidfを計算")
    tfidf_array = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        tfidf_array.append(get_tfidf(row))
    tfidf_array = np.array(tfidf_array).squeeze()
    print(f"tfidf_array:{tfidf_array.shape}")

    # dfの各行について、A,B,C,D,Eのカラムそれぞれを別の行にする
    print("A,B,C,D,Eのカラムを分割")
    records = []
    for di, row in df.iterrows():
        for oi, option in enumerate("ABCDE"):
            record = {
                "df_index": di,
                "option_index": oi,
                "option": option,
                "label": option == row["answer"],
                "context_len": len(row["context"].split(" ")),
                "prompt_len": len(row["prompt"].split(" ")),
                "option_len": len(row[option].split(" ")),
                "pred": pred[di, oi],
                "pred_div": pred[di, oi] / pred[di].max(),
                "pred_max": pred[di].max(),
                "pred_min": pred[di].min(),
                "pred_mean": pred[di].mean(),
                "pred_std": pred[di].std(),
                "sim_max": row["sim_max"],
                "sim_min": row["sim_min"],
                "sim_mean": row["sim_mean"],
                "sim_std": row["sim_std"],
                "sim_num": row["sim_num"],
                "tfidf": tfidf_array[di, oi],
                "tfidf_div": tfidf_array[di, oi] / tfidf_array[di].max(),
                "tfidf_max": tfidf_array[di].max(),
                "tfidf_min": tfidf_array[di].min(),
                "tfidf_mean": tfidf_array[di].mean(),
                "tfidf_std": tfidf_array[di].std(),
            }
            print(record)
            records.append(record)

    # 作成したデータをDataFrameに変換
    dataset_df = pd.DataFrame(records)
    return dataset_df


# hydraで設定を読み込む
@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c: DictConfig) -> None:
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.preprocess

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.preprocess.split('/')[-1]}"
    preprocessed_path = Path(f"./preprocessed/{exp_name}")

    for data_name, data_dict in OmegaConf.to_container(cfg)["data"].items():
        print(f"preprocess {data_name}")
        print(data_dict)
        df = pd.read_csv(data_dict["base_path"])
        pred = np.load(data_dict["pred_path"])
        print(f"df:{df.shape}, pred:{pred.shape}")
        dataset_df = make_dataset(df, pred)
        dataset_df.to_csv(preprocessed_path / f"{data_name}_2nd.csv", index=False)


if __name__ == "__main__":
    main()
