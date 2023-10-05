"""
1stの結果を読み込んで、確率値の最大値を予測とする。
また、tfidfベースの後処理によって、予測結果を修正する（具体的にはレーベンシュタイン距離がほぼ変わらない選択肢同士の場合、tfidfのスコアに応じて予測結果を修正する）。

"""


import os
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

import Levenshtein
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

stop_words = list(stopwords.words("english"))


def get_tfidf(row: dict[str, str]) -> np.ndarray:
    """
    tfidfを計算する。大きめのngramにすることでうまく計算できるようにする
    """
    # tfidfの計算
    tfidf = TfidfVectorizer(
        ngram_range=(3, 7),
        # token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
        stop_words=stop_words,
    )
    base_cols = ["A", "B", "C", "D", "E"]
    fit_cols = base_cols + ["context", "prompt"]
    tfidf_vec = tfidf.fit([row[col] for col in fit_cols])
    # base_cols と context の 類似度を計算
    base_vec = tfidf_vec.transform([row[col] for col in base_cols])
    context_vec = tfidf_vec.transform([row["context"]])
    sim = cosine_similarity(base_vec, context_vec)
    return sim


def add_feat_by_prob(df, max_prob):
    first_prob = np.sort(max_prob)[:, -1]
    second_prob = np.sort(max_prob)[:, -2]
    third_prob = np.sort(max_prob)[:, -3]
    prob_diff = first_prob - second_prob
    df["first_prob"] = first_prob
    df["second_prob"] = second_prob
    df["third_prob"] = third_prob
    df["prob_diff"] = prob_diff
    df["prob_diff23"] = second_prob - third_prob

    option_to_index = {option: idx for idx, option in enumerate("ABCDE")}
    index_to_option = {v: k for k, v in option_to_index.items()}
    first_option = np.argsort(max_prob)[:, -1]
    df["first_option_index"] = first_option
    df["first_option"] = df["first_option_index"].map(index_to_option)
    second_option = np.argsort(max_prob)[:, -2]
    df["second_option_index"] = second_option
    df["second_option"] = df["second_option_index"].map(index_to_option)
    third_option = np.argsort(max_prob)[:, -3]
    df["third_option_index"] = third_option
    df["third_option"] = df["third_option_index"].map(index_to_option)

    df["first_len"] = 0
    df["second_len"] = 0
    df["third_len"] = 0
    for i, row in df.iterrows():
        df.loc[i, "first_len"] = len(row[row["first_option"]])
        df.loc[i, "second_len"] = len(row[row["second_option"]])
        df.loc[i, "third_len"] = len(row[row["third_option"]])

    # first と second のレーベンシュタイン距離
    dists = []
    for i, row in tqdm(df.iterrows()):
        dists.append(Levenshtein.distance(row[row["first_option"]], row[row["second_option"]]))
    df["dist_1_2"] = dists
    df["dist_1_2_rate"] = df["dist_1_2"] / df[["first_len", "second_len"]].max(axis=1)

    # 正解がfirst, second, other のどれかを見る
    df["answer_location"] = "other"
    df.loc[df["first_option"] == df["answer"], "answer_location"] = "first"
    df.loc[df["second_option"] == df["answer"], "answer_location"] = "second"
    df.loc[df["third_option"] == df["answer"], "answer_location"] = "third"

    print("tfidfを計算")
    tfidf_array = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        tfidf_array.append(get_tfidf(row).squeeze())
    tfidf_array = np.array(tfidf_array)
    print(f"tfidf_array:{tfidf_array.shape}")

    df["first_tfidf"] = tfidf_array[np.arange(len(df)), first_option]
    df["second_tfidf"] = tfidf_array[np.arange(len(df)), second_option]

    df["should_swap"] = (
        (df["prob_diff"] < 0.3)
        & (df["dist_1_2_rate"] < 0.2)
        & (20 < df["first_len"])
        & (0.001 < df["first_tfidf"])
        & (df["second_tfidf"] / df["first_tfidf"] > 1.4)
    )
    return df


def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k


def map_k(true_items, predictions, K=3):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_k = 0.0
    for u in range(U):
        user_preds = predictions[u]
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), K)):
            map_at_k += precision_at_k(user_results, k + 1) * user_results[k]
    return map_at_k / U


def mistake_idx(true_items, predictions):
    """
    trueとpredのtop1がそれぞれ違うidxを返す
    """
    U = len(predictions)
    mistake_idx = []
    for u in range(U):
        user_preds = predictions[u]
        user_true = true_items[u]
        if user_preds[0] != user_true:
            mistake_idx.append(u)
    return mistake_idx


option_to_index = {option: idx for idx, option in enumerate("ABCDE")}
index_to_option = {v: k for k, v in option_to_index.items()}


def predictions_to_map_output(predictions):
    sorted_answer_indices = np.argsort(-predictions)  # Sortting indices in descending order
    top_answer_indices = sorted_answer_indices[:, :]  # Taking the first three indices for each row
    top_answers = np.vectorize(index_to_option.get)(
        top_answer_indices
    )  # Transforming indices to options - i.e., 0 --> A
    return np.apply_along_axis(lambda row: " ".join(row), 1, top_answers)


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c: DictConfig) -> None:
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.exp

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.exp.split('/')[-1]}"
    output_path = Path(f"./output/{exp_name}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(cfg)

    # data1 = pd.read_csv(cfg.data1_path)
    data2 = pd.read_csv(cfg.tfidf_path)

    # numpy の予測結果を読み込む
    # pred1_list = []
    pred2_list = []
    for dir_path in cfg.pred_dirs:
        # pred1_list.append(np.load(f"{dir_path}/data1_pred.npy"))
        pred2_list.append(np.load(f"{dir_path}/data2_pred.npy"))

    # 確率値のmaxを取って、予測結果を作成
    # pred1 = np.max(pred1_list, axis=0)
    pred2 = np.max(pred2_list, axis=0)
    np.save(output_path / "data2_pred.npy", pred2)

    data2 = add_feat_by_prob(data2, pred2)

    # before post process
    out2 = predictions_to_map_output(pred2)
    true2 = data2["answer"].values
    map2 = map_k(true2, out2)
    map3 = map_k(true2[:200], out2[:200])
    print(f"map2:{map2}, map3:{map3}")
    print("mistake_idx2:", mistake_idx(true2, out2))
    print()

    # post process (swap すべきところをswapするために、予測結果を修正する)
    print("post process")
    pred2[data2["should_swap"], data2.loc[data2["should_swap"], "second_option_index"]] = (
        pred2[data2["should_swap"], data2.loc[data2["should_swap"], "first_option_index"]] + 1.0
    )
    out2 = predictions_to_map_output(pred2)
    true2 = data2["answer"].values
    map2 = map_k(true2, out2)
    map3 = map_k(true2[:200], out2[:200])
    print(f"map2:{map2}, map3:{map3}")


if __name__ == "__main__":
    main()
