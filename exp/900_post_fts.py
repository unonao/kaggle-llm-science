"""
1stの結果を読み込んで、確率値の最大値を予測とする。
また、tfidfベースの後処理によって、予測結果を修正する（具体的にはレーベンシュタイン距離がほぼ変わらない選択肢同士の場合、tfidfのスコアに応じて予測結果を修正する）。

"""


import os
import time
import re
import sys
from pathlib import Path
from dataclasses import dataclass

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from typing import Optional, Union
import torch
from datasets import Dataset

import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import sqlite3
from transformers import (
    AutoModel,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

from nltk.corpus import stopwords
from sklearn.feature_extraction import text

stop_words = list(stopwords.words("english"))
print(len(stop_words))
stop_words2 = text.ENGLISH_STOP_WORDS
stop_words = list(stop_words2.union(stop_words))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


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
        (df["prob_diff"] < 0.2)
        & (df["dist_1_2_rate"] < 0.5)
        & (20 < df["first_len"])
        & ((0.01 < df["first_tfidf"]) | (0.01 < df["second_tfidf"]))
        & (df["second_tfidf"] / (df["first_tfidf"] + 1e-9) > 1.2)
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


def search_fts(cur, row, num_sentences=10):
    text = row.prompt + " " + row.A + " " + row.B + " " + row.C + " " + row.D + " " + row.E
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(2, 2), stop_words=stop_words, max_features=20)
    vectorizer.fit([text])
    features = list(vectorizer.get_feature_names_out())
    q = " OR ".join(["(" + " AND ".join(text.split()) + ")" for text in features])
    res = cur.execute(
        f"""select text
            from imdb
            where text MATCH "{q}"
            ORDER BY rank
            limit {num_sentences}""",
    ).fetchall()
    if len(res) == 0:
        return ""
    else:
        # 結果を一つに結合
        res = [r[0][:2000] for r in res]
        res = " ".join(res)
        # 一定の長さにする
        res = res[:5000]
        return res


def predict_from_df(df: pd.DataFrame, model_path: str) -> np.ndarray:
    """
    dfから予測する
    """
    df = df.copy()

    def clean_text(text):
        text = text.replace('"', "")
        text = text.replace("“", "")
        text = text.replace("”", "")
        return text

    def preprocess_df(df):
        df["prompt_with_context"] = (
            df["context"].fillna("no context").apply(lambda x: " ".join(x.split()[:300]))
            + f"... ['SEP'] "
            + df["prompt"].fillna("")
        )
        df["prompt_with_context"] = df["prompt_with_context"].apply(clean_text)
        options = ["A", "B", "C", "D", "E"]
        for option in options:
            df[option] = df[option].fillna("")
        return df

    df = preprocess_df(df)
    dataset_valid = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    option_to_index = {option: idx for idx, option in enumerate("ABCDE")}
    index_to_option = {v: k for k, v in option_to_index.items()}

    def preprocess(example):
        first_sentence = [example["prompt_with_context"]] * 5
        second_sentences = [example[option] for option in "ABCDE"]
        tokenized_example = tokenizer(first_sentence, second_sentences, truncation=False)
        tokenized_example["label"] = 0  # option_to_index[example["answer"]]
        return tokenized_example

    tokenized_dataset_valid = dataset_valid.map(
        preprocess, batched=False, remove_columns=["prompt_with_context", "prompt", "A", "B", "C", "D", "E", "answer"]
    )
    model = AutoModelForMultipleChoice.from_pretrained(model_path)
    args = TrainingArguments(output_dir="output/tmp", per_device_eval_batch_size=1)
    trainer = Trainer(
        model=model, args=args, tokenizer=tokenizer, data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer)
    )
    valid_pred = trainer.predict(tokenized_dataset_valid).predictions
    return torch.softmax(torch.tensor(valid_pred), dim=1).numpy()


def modify_prob_by_fts(
    df: pd.DataFrame, prob: np.ndarray, fts_time_limit: int, model_path: str, fts_path: str
) -> tuple[pd.DataFrame, np.ndarray]:
    # cfg.fts_time_limit の間だけ、np.max(pred2,axis=1) の値が小さいものから順に、FTSでcontextを修正し、予測し直して確率値を修正する
    db = sqlite3.connect(fts_path)
    cur = db.cursor()

    max_prob = np.max(prob, axis=1)
    sorted_idx = np.argsort(max_prob)

    ftx_index_list = []  # あとで予測し直して確率値を修正するindexを保存する

    start_time = time.time()
    for idx in tqdm(sorted_idx):
        if time.time() - start_time > fts_time_limit:
            break
        row = df.iloc[idx]
        # FTSでcontextを修正
        new_context = search_fts(cur, row)
        if len(new_context) == 0:
            continue
        ftx_index_list.append(idx)
        df.loc[idx, "context"] = new_context  # contextを修正
    partial_df = df.iloc[ftx_index_list]
    partial_pred = predict_from_df(partial_df, model_path)
    print(ftx_index_list)
    prob[ftx_index_list] = np.max([prob[ftx_index_list], partial_pred], axis=0)
    return df, prob


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

    # debug
    if cfg.debug:
        data2 = data2.head(10)
        pred2 = pred2[:10]

    # before post process
    out2 = predictions_to_map_output(pred2)
    true2 = data2["answer"].values
    map2 = map_k(true2, out2)
    map3 = map_k(true2[:200], out2[:200])
    print(f"map2:{map2}, map3:{map3}")
    print("mistake_idx2:", mistake_idx(true2, out2))
    print()

    # after post process
    # FTSによる修正
    print("##### FTS #####")
    data2, pred2 = modify_prob_by_fts(
        df=data2, prob=pred2, fts_time_limit=cfg.fts_time_limit, model_path=cfg.model_path, fts_path=cfg.fts_path
    )
    out2 = predictions_to_map_output(pred2)
    true2 = data2["answer"].values
    map2 = map_k(true2, out2)
    map3 = map_k(true2[:200], out2[:200])
    print(f"map2:{map2}, map3:{map3}")
    print("mistake_idx2:", mistake_idx(true2, out2))
    print()

    # after post process2
    print("##### post process #####")
    data2 = add_feat_by_prob(data2, pred2)
    pred2[data2["should_swap"], data2.loc[data2["should_swap"], "second_option_index"]] = (
        pred2[data2["should_swap"], data2.loc[data2["should_swap"], "first_option_index"]] + 1.0
    )
    out2 = predictions_to_map_output(pred2)
    true2 = data2["answer"].values
    map2 = map_k(true2, out2)
    map3 = map_k(true2[:200], out2[:200])
    print(f"map2:{map2}, map3:{map3}")
    print("mistake_idx2:", mistake_idx(true2, out2))


if __name__ == "__main__":
    main()
