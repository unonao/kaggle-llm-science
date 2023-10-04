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

import blingfire as bf
import faiss
import hydra
import numpy as np
import pandas as pd
import torch
from faiss import read_index, write_index
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

libc = ctypes.CDLL("libc.so.6")
sys.path.append(os.pardir)

import utils


def extract_chunk_by_sliding_window(text_list: list[str], window_size: int, sliding_size: int) -> list[str]:
    """
    text のリストをsliding windowで結合する。window_size個のtextが含まれるまで結合し、sliding_size個ずつずらして結合する。
    """
    chunks = []
    for i in range(0, len(text_list), sliding_size):
        chunk = " ".join(text_list[i : i + window_size])
        chunks.append(chunk)
    return chunks


def extract_sections(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"#{2,}\s?(.*?)\s?#{2,}")
    sections = []

    matches = list(pattern.finditer(text))
    start_idx = 0

    if len(matches) == 0:
        sections.append(("", text))
        return sections

    for i, match in enumerate(matches):
        if i == 0:
            end_idx = match.start()
            sections.append(("", text[start_idx:end_idx].strip()))

        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = (match.group(1).strip(), text[start_idx:end_idx].strip())
        if section[0] not in ["See also", "References", "Further reading", "External links"]:
            sections.append(section)

        start_idx = end_idx

    # 空のtextの場合は飛ばす
    sections = [section for section in sections if len(section[1].split(" ")) >= 3]
    return sections


def sentencize(
    titles: Iterable[str],
    documents: Iterable[str],
    document_ids: Iterable,
    window_size: int = 3,
    sliding_size: int = 2,
    filter_len: int = 5,
    filter_len_max: int = 500,
    disable_progress_bar: bool = False,
) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents`
    to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the
    location in the original documents.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    document_sentences = []
    for title, document, document_id in tqdm(
        zip(titles, documents, document_ids), total=len(documents), disable=disable_progress_bar
    ):
        try:
            # chunk にまとめる
            ## 念のため改行をスペースに変換
            document = document.replace("\n", " ")
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            section_sentences = []
            for o in sentence_offsets:
                if filter_len < o[1] - o[0] and o[1] - o[0] < filter_len_max:
                    section_sentences.append(document[o[0] : o[1]])
            chunks = extract_chunk_by_sliding_window(section_sentences, window_size, sliding_size)

            for chunk in chunks:
                row = {}
                row["document_id"] = document_id
                row["text"] = f"{title} > {chunk}"
                row["offset"] = (0, 0)
                document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)


def sectionize_documents(
    titles: Iterable[str],
    documents: Iterable[str],
    document_ids: Iterable,
    disable_progress_bar: bool = False,
) -> pd.DataFrame:
    """
    Obtains the sections of the imaging reports and returns only the
    selected sections (defaults to FINDINGS, IMPRESSION, and ADDENDUM).

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `offset`
    """
    processed_documents = []
    for title, document_id, document in tqdm(
        zip(titles, document_ids, documents), total=len(documents), disable=disable_progress_bar
    ):
        row = {}
        text, start, end = (document, 0, len(document))
        row["document_id"] = document_id
        row["text"] = text
        row["offset"] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(["document_id", "offset"]).reset_index(drop=True)
    else:
        return _df


def relevant_title_retrieval(
    df: pd.DataFrame,
    index_path: str,
    model: SentenceTransformer,
    top_k: int = 3,
    batch_size: int = 32,
) -> pd.DataFrame:
    sentence_index = read_index(index_path)  # index 読み込み
    res = faiss.StandardGpuResources()  # use a single GPU
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    # sentence_index = faiss.index_cpu_to_gpu(res, 0, sentence_index, co)
    sentence_index.nprobe = 10
    prompt_embeddings = model.encode(
        df.prompt_answer_stem.values,
        batch_size=batch_size,
        device="cuda",
        show_progress_bar=True,
        # convert_to_tensor=True,
        normalize_embeddings=True,
    )
    prompt_embeddings = prompt_embeddings.astype(np.float32)
    # prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
    search_score, search_index = sentence_index.search(prompt_embeddings, top_k)
    res.noTempMemory()
    del res
    del sentence_index
    del prompt_embeddings
    _ = gc.collect()
    libc.malloc_trim(0)
    return search_score, search_index


def get_wikipedia_file_data(
    search_score: np.ndarray,
    search_index: np.ndarray,
    wiki_index_path: str,
) -> pd.DataFrame:
    wiki_index_df = pd.read_parquet(wiki_index_path, columns=["id", "file"])
    wikipedia_file_data = []
    for i, (scr, idx) in tqdm(enumerate(zip(search_score, search_index)), total=len(search_score)):
        _df = wiki_index_df.loc[idx].copy()
        _df["prompt_id"] = i
        wikipedia_file_data.append(_df)
    wikipedia_file_data = pd.concat(wikipedia_file_data).reset_index(drop=True)
    wikipedia_file_data = (
        wikipedia_file_data[["id", "prompt_id", "file"]]
        .drop_duplicates()
        .sort_values(["file", "id"])
        .reset_index(drop=True)
    )
    ## Save memory - delete df since it is no longer necessary
    del wiki_index_df
    _ = gc.collect()
    libc.malloc_trim(0)
    return wikipedia_file_data


def get_full_text_data(
    wikipedia_file_data: pd.DataFrame,
    wiki_dir: str,
):
    ## Get the full text data
    wiki_text_data = []
    for file in tqdm(wikipedia_file_data.file.unique(), total=len(wikipedia_file_data.file.unique())):
        _id = [str(i) for i in wikipedia_file_data[wikipedia_file_data["file"] == file]["id"].tolist()]
        _df = pd.read_parquet(f"{wiki_dir}/{file}", columns=["id", "title", "text"])
        _df_temp = _df[_df["id"].isin(_id)].copy()
        del _df
        _ = gc.collect()
        libc.malloc_trim(0)
        wiki_text_data.append(_df_temp)
    wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
    _ = gc.collect()
    libc.malloc_trim(0)
    return wiki_text_data


def extract_contexts_from_matching_pairs(
    df: pd.DataFrame,
    processed_wiki_text_data: pd.DataFrame,
    wikipedia_file_data: pd.DataFrame,
    wiki_data_embeddings: np.ndarray,
    question_embeddings: np.ndarray,
    num_sentences_include: int = 5,
):
    results = {"contexts": [], "sim_min": [], "sim_max": [], "sim_mean": [], "sim_std": [], "sim_num": []}
    for r in tqdm(df.itertuples(), total=len(df)):
        prompt_id = r.Index
        prompt_indices = processed_wiki_text_data[
            processed_wiki_text_data["document_id"].isin(
                wikipedia_file_data[wikipedia_file_data["prompt_id"] == prompt_id]["id"].values
            )
        ].index.values
        assert prompt_indices.shape[0] > 0
        prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
        prompt_index.add(wiki_data_embeddings[prompt_indices])
        ## Get the top matches
        ss, ii = prompt_index.search(question_embeddings[np.newaxis, prompt_id], num_sentences_include)
        context = ""
        total_len = 0
        num = 0
        for _s, _i in zip(ss[0], ii[0]):
            if total_len > 1000 or _s >= 1.0:
                break
            text = processed_wiki_text_data.loc[prompt_indices]["text"].iloc[_i]
            context += text + " "
            total_len += len(text.split(" "))
            num += 1
        results["contexts"].append(context)
        results["sim_max"].append(ss[0][:num].max())
        results["sim_min"].append(ss[0][:num].min())
        results["sim_mean"].append(ss[0][:num].mean())
        results["sim_std"].append(ss[0][:num].std())
        results["sim_num"].append(num)

    return results


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

    # モデル読み込み
    model = SentenceTransformer(cfg.sim_model, device="cuda")
    model.max_seq_length = cfg.max_length
    model = model.half()

    for path in cfg.data_paths:
        # データ読み込み
        df = pd.read_csv(path)
        df[["A", "B", "C", "D", "E"]] = df[["A", "B", "C", "D", "E"]].fillna("")

        df.reset_index(inplace=True, drop=True)
        if cfg.debug:
            df = df.head(15)
        print(f"{path}:{df.shape}")
        df["answer_all"] = df.apply(lambda x: " ".join([x["A"], x["B"], x["C"], x["D"], x["E"]]), axis=1)
        df["prompt_answer_stem"] = df["prompt"] + " " + df["answer_all"]
        df["prompt_answer_stem"] = df["prompt_answer_stem"].str.replace('"', "")
        df["prompt_answer_stem"] = df["prompt_answer_stem"].str.replace("“", "")
        df["prompt_answer_stem"] = df["prompt_answer_stem"].str.replace("”", "")

        # title 検索
        print("【title 検索】")
        search_score, search_index = relevant_title_retrieval(
            df,
            cfg.index_path,
            model,
            top_k=cfg.doc_top_k,
            batch_size=cfg.batch_size,
        )

        # wikipedia file data 取得 ("id", "prompt_id", "file")
        print("【wikipedia file data 取得】")
        wikipedia_file_data = get_wikipedia_file_data(
            search_score,
            search_index,
            cfg.wiki_index_path,
        )
        print(wikipedia_file_data.head())
        del search_score
        del search_index
        _ = gc.collect()

        # wikipedia text data 取得 ("id", "title", "text")
        print("【wikipedia text data 取得】")
        wiki_text_data = get_full_text_data(
            wikipedia_file_data,
            cfg.wiki_dir,
        )
        print(wiki_text_data.tail())

        ## Parse documents into sentences
        print("【sentencize】")
        processed_wiki_text_data = sentencize(
            wiki_text_data.title.values,
            wiki_text_data.text.values,
            wiki_text_data.id.values,
            cfg.window_size,
            cfg.sliding_size,
        )
        print(processed_wiki_text_data.tail())
        # print data size(GB) of processed_wiki_text_data
        print("processed_wiki_text_data size(GB):", sys.getsizeof(processed_wiki_text_data) / 1024**3)
        del wiki_text_data  # 追加
        _ = gc.collect()

        ## Get embeddings of the wiki text data
        print("【Get embeddings of the wiki text data】")
        wiki_data_embeddings = model.encode(
            processed_wiki_text_data.text,
            batch_size=cfg.batch_size,
            device="cuda",
            show_progress_bar=True,
            # convert_to_tensor=True,
            normalize_embeddings=True,
        )  # .half()
        # wiki_data_embeddings = wiki_data_embeddings.detach().cpu().numpy()
        wiki_data_embeddings = wiki_data_embeddings.astype(np.float32)
        # print data size(GB) of wiki_data_embeddings
        print("wiki_data_embeddings size(GB):", sys.getsizeof(wiki_data_embeddings) / 1024**3)
        print(wiki_data_embeddings[0, :10])

        _ = gc.collect()
        torch.cuda.empty_cache()

        ## Combine all answers
        print("【Combine all answers】")
        question_embeddings = model.encode(
            df.prompt_answer_stem.values,
            batch_size=cfg.batch_size,
            device="cuda",
            show_progress_bar=True,
            # convert_to_tensor=True,
            normalize_embeddings=True,
        )
        question_embeddings = question_embeddings.astype(np.float32)
        df.drop(["answer_all", "prompt_answer_stem"], axis=1, inplace=True)
        # question_embeddings = question_embeddings.detach().cpu().numpy()
        torch.cuda.empty_cache()
        print(question_embeddings[0, :10])

        ## Extract contexts from matching pairs
        print("【Extract contexts from matching pairs】")
        results = extract_contexts_from_matching_pairs(
            df,
            processed_wiki_text_data,
            wikipedia_file_data,
            wiki_data_embeddings,
            question_embeddings,
            num_sentences_include=cfg.num_sentences_include,
        )
        df["context"] = results["contexts"]
        df["sim_max"] = results["sim_max"]
        df["sim_min"] = results["sim_min"]
        df["sim_mean"] = results["sim_mean"]
        df["sim_std"] = results["sim_std"]
        df["sim_num"] = results["sim_num"]

        # 保存
        print("【保存】")
        preprocessed_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(preprocessed_path / f"{Path(path).stem}.csv", index=False)

        del wiki_data_embeddings
        del question_embeddings
        del df
        del wikipedia_file_data
        del processed_wiki_text_data
        _ = gc.collect()


if __name__ == "__main__":
    main()
