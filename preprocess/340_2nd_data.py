"""
1st の学習結果を使って 2nd のデータを作成する

1. 元データと1stの結果読み込み
2. tf-idf なども用いて特徴量を作成。集約特徴量も作成
3. (prompt_id, option) で一意の行になるように分割
"""

import pandas as pd
import numpy as np

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_tfidf(row: dict[str, str]):
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


# hydraで設定を読み込む
@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c: DictConfig) -> None:
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.preprocess

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.preprocess.split('/')[-1]}"
    preprocessed_path = Path(f"./preprocessed/{exp_name}")

    for data_name, data_dict in cfg.data:
        df = pd.read_csv(data_dict.base_path)
        pred = np.load(data_dict.pred_path)
        dataset_df = make_dataset(df, pred)
        dataset_df.to_csv(preprocessed_path / f"{data_name}_2nd.csv", index=False)
