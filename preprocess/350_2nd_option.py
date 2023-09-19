"""
option同士での平均距離を特徴量に追加
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
import re


def clean_text(text: str) -> str:
    """
    tf-idfの前にテキストを整形する
    """
    text = text.lower()
    text = re.sub(r'[?|!|\'|"|#]', r"", text)
    text = re.sub(r"[.|,|)|(|\|/]", r" ", text)
    return text


def get_tfidf(row: dict[str, str]) -> np.ndarray:
    """
    tfidfを計算する
    """
    # tfidfの計算
    tfidf = TfidfVectorizer()
    base_cols = ["A", "B", "C", "D", "E"]
    fit_cols = base_cols + ["prompt"]
    tfidf_vec = tfidf.fit([row[col] for col in fit_cols])
    # base_cols の 類似度を計算
    base_vec = tfidf_vec.transform([row[col] for col in base_cols])
    sim_options = cosine_similarity(base_vec, base_vec)
    mean_sim_options = sim_options.mean(axis=1)
    return mean_sim_options


def make_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    元データと1stの予測結果を結合して2ndのデータを作成する
    """
    # テキスト整形
    print("テキスト整形")
    for col in ["context", "prompt", "A", "B", "C", "D", "E"]:
        df[col] = df[col].fillna("").apply(clean_text)

    # まずはtfidf
    print("tfidfを計算")
    tfidf_option_array = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sim_options = get_tfidf(row)
        tfidf_option_array.append(sim_options)

    tfidf_option_array = np.array(tfidf_option_array).squeeze()
    print(f"tfidf_option_array:{tfidf_option_array.shape}")

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
                "len_prompt": len(row["prompt"].split(" ")),
                "len_option": len(row[option].split(" ")),
                "len_sum": len(row["prompt"].split(" ")) + len(row[option].split(" ")),
                "tfidf_option_mean": tfidf_option_array[di, oi],
                "tfidf_option_div": tfidf_option_array[di, oi] / (tfidf_option_array[di].max() + 1e-8),
                "tfidf_option_max": tfidf_option_array[di].max(),
                "tfidf_option_min": tfidf_option_array[di].min(),
                "tfidf_option_mean": tfidf_option_array[di].mean(),
                "tfidf_option_std": tfidf_option_array[di].std(),
            }
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
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    for data_name, data_dict in OmegaConf.to_container(cfg)["data"].items():
        print(f"preprocess {data_name}")
        print(data_dict)
        df = pd.read_csv(data_dict["base_path"])
        print(f"df:{df.shape},")
        dataset_df = make_dataset(df)
        dataset_df.to_csv(preprocessed_path / f"{data_name}_2nd.csv", index=False)


if __name__ == "__main__":
    main()
