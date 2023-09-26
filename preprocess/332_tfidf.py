"""
wikipedia から質問文に関係する文章を抽出して preprocessed に保存するスクリプト
"""

from __future__ import annotations

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
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk

from nltk.corpus import stopwords

stop_words = list(stopwords.words("english"))  # 使用言語に応じて変更
libc = ctypes.CDLL("libc.so.6")
sys.path.append(os.pardir)

import utils


def SplitList(mylist, chunk_size):
    return [mylist[offs : offs + chunk_size] for offs in range(0, len(mylist), chunk_size)]


def get_relevant_documents_parsed(df_valid, debug=False):
    df_chunk_size = 4000
    paraphs_parsed_dataset = load_from_disk("input/all-paraphs-parsed-expanded")
    if debug:
        paraphs_parsed_dataset = paraphs_parsed_dataset.select(range(1000))
    modified_texts = paraphs_parsed_dataset.map(
        lambda example: {
            "temp_text": f"{example['title']} {example['section']} {example['text']}".replace("\n", " ").replace(
                "'", ""
            )
        },
        num_proc=4,
    )["temp_text"]

    all_articles_indices = []
    all_articles_values = []
    for idx in tqdm(range(0, df_valid.shape[0], df_chunk_size)):
        df_valid_ = df_valid.iloc[idx : idx + df_chunk_size]

        articles_indices, merged_top_scores = retrieval(df_valid_, modified_texts)
        all_articles_indices.append(articles_indices)
        all_articles_values.append(merged_top_scores)

    article_indices_array = np.concatenate(all_articles_indices, axis=0)
    articles_values_array = np.concatenate(all_articles_values, axis=0).reshape(-1)

    top_per_query = article_indices_array.shape[1]
    articles_flatten = [
        (
            articles_values_array[index],
            str(paraphs_parsed_dataset[idx.item()]["title"])
            + " > "
            + str(paraphs_parsed_dataset[idx.item()]["section"])
            + " > "
            + str(paraphs_parsed_dataset[idx.item()]["text"]),
        )
        for index, idx in enumerate(article_indices_array.reshape(-1))
    ]
    retrieved_articles = SplitList(articles_flatten, top_per_query)

    articles = []
    for retrieved_articles_per_query in retrieved_articles:
        context = ""
        for i in range(1, top_per_query):
            context += retrieved_articles_per_query[-i][1] + " "
        articles.append(context.replace("\n", " "))
    return articles


def retrieval(df_valid, modified_texts):
    corpus_df_valid = df_valid.apply(
        lambda row: f'{row["prompt"]}\n{row["prompt"]}\n{row["prompt"]}\n{row["A"]}\n{row["B"]}\n{row["C"]}\n{row["D"]}\n{row["E"]}',
        axis=1,
    ).values
    vectorizer1 = TfidfVectorizer(
        ngram_range=(1, 2), token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'", stop_words=stop_words
    )
    vectorizer1.fit(corpus_df_valid)
    vocab_df_valid = vectorizer1.get_feature_names_out()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
        stop_words=stop_words,
        vocabulary=vocab_df_valid,
    )
    vectorizer.fit(modified_texts[:500000])
    corpus_tf_idf = vectorizer.transform(corpus_df_valid)

    print(f"length of vectorizer vocab is {len(vectorizer.get_feature_names_out())}")

    chunk_size = 100000
    top_per_chunk = 10
    top_per_query = 10

    all_chunk_top_indices = []
    all_chunk_top_values = []

    for idx in tqdm(range(0, len(modified_texts), chunk_size)):
        wiki_vectors = vectorizer.transform(modified_texts[idx : idx + chunk_size])
        temp_scores = (corpus_tf_idf * wiki_vectors.T).toarray()
        chunk_top_indices = temp_scores.argpartition(-top_per_chunk, axis=1)[:, -top_per_chunk:]
        chunk_top_values = temp_scores[np.arange(temp_scores.shape[0])[:, np.newaxis], chunk_top_indices]

        all_chunk_top_indices.append(chunk_top_indices + idx)
        all_chunk_top_values.append(chunk_top_values)

    top_indices_array = np.concatenate(all_chunk_top_indices, axis=1)
    top_values_array = np.concatenate(all_chunk_top_values, axis=1)

    merged_top_scores = np.sort(top_values_array, axis=1)[:, -top_per_query:]
    merged_top_indices = top_values_array.argsort(axis=1)[:, -top_per_query:]
    articles_indices = top_indices_array[np.arange(top_indices_array.shape[0])[:, np.newaxis], merged_top_indices]

    return articles_indices, merged_top_scores


# hydraで設定を読み込む
@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c: DictConfig) -> None:
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.preprocess

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.preprocess.split('/')[-1]}"
    preprocessed_path = Path(f"./preprocessed/{exp_name}")

    print(cfg)
    print("preprocessed_path:", preprocessed_path)

    for path in cfg.data_paths:
        # データ読み込み
        df = pd.read_csv(path)
        df[["A", "B", "C", "D", "E"]] = df[["A", "B", "C", "D", "E"]].fillna("")

        df.reset_index(inplace=True, drop=True)
        if cfg.debug:
            df = df.head(15)
        print(f"{path}:{df.shape}")

        retrieved_articles_parsed = get_relevant_documents_parsed(df, cfg.debug)
        df["context"] = retrieved_articles_parsed

        # 保存
        print("【保存】")
        preprocessed_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(preprocessed_path / f"{Path(path).stem}.csv", index=False)


if __name__ == "__main__":
    main()
