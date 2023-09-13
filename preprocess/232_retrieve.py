"""
wikipedia-2023-07-faiss-index, wikipedia-20230701 を使って、
wikipedia から質問文に関係する文章を抽出してpreprocessed に保存するスクリプト
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


def split_paragraph(paragraph: str, max_paragraph_length: int = 100) -> list[str]:
    # paragraphをちょうどよい単位に分割する

    _, sentence_offsets = bf.text_to_sentences_and_offsets(paragraph)
    buffer = ""
    paragraphs = []
    for o in sentence_offsets:
        sentence = paragraph[o[0] : o[1]]

        if len((buffer + sentence).split(" ")) <= max_paragraph_length:
            buffer += sentence + " "
        else:
            paragraphs.append(buffer.strip())
            buffer = sentence + " "

    if buffer:
        paragraphs.append(buffer.strip())

    # 空のセクションをフィルタリング
    chunks = [chunk for chunk in paragraphs if len(chunk) > 0]
    return chunks


def sentencize(
    documents: Iterable[str],
    document_ids: Iterable,
    offsets: Iterable[tuple[int, int]],
    filter_len: int = 10,
    disable_progress_bar: bool = False,
) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents`
    to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the
    location in the original documents.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param offsets: Iterable tuple of the start and end indices
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    document_sentences = []
    for document, document_id, offset in tqdm(
        zip(documents, document_ids, offsets), total=len(documents), disable=disable_progress_bar
    ):
        # 不要と思われる部分は削除する
        document = document.split("##See also##")[0]
        document = document.split("## See also ##")[0]
        document = document.split("##References##")[0]
        document = document.split("## References ##")[0]
        document = document.split("##Further reading##")[0]
        document = document.split("## Further reading ##")[0]
        document = document.split("##External links##")[0]
        document = document.split("## External links ##")[0]

        # パターンを : に変換
        pattern = re.compile(r"#{2,}\s?(.*?)\s?#{2,}(\n)*")
        document = pattern.sub(r" \1 : ", document)

        # formula_1, formula_11 などを変換
        document = re.sub(r"formula_\d+", "Formula", document)

        pattern = re.compile(r"&lt;templatestyles.*/&gt;")
        document = pattern.sub(r"", document)
        pattern = re.compile(r"&lt;br&gt;")
        document = document.replace('"', "")

        # " を削除
        document = document.replace('"', "")
        document = document.replace("“", "")
        document = document.replace("”", "")

        # document = document.replace("\n\n", " ")
        # document = document.replace("\n", "<next>")  # これで前後が強制的につながる
        # document = document.replace("\n", "")

        try:
            for paragraph in document.split("\n"):
                for sentence in split_paragraph(paragraph):
                    if len(sentence) > filter_len:
                        row = {}
                        row["document_id"] = document_id
                        row["text"] = sentence
                        # <next> を空白に置換
                        # row["text"] = row["text"].replace("<next>", " ")
                        # row["text"] = row["text"].replace("\n", " ")
                        row["offset"] = (0, 0)
                        document_sentences.append(row)
            """
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1] - o[0] > filter_len:
                    sentence = document[o[0] : o[1]]
                    abs_offsets = (o[0] + offset[0], o[1] + offset[0])
                    row = {}
                    row["document_id"] = document_id
                    row["text"] = sentence
                    # <next> を空白に置換
                    # row["text"] = row["text"].replace("<next>", " ")
                    # row["text"] = row["text"].replace("\n", " ")
                    row["offset"] = abs_offsets
                    document_sentences.append(row)
            """
        except:
            continue
    return pd.DataFrame(document_sentences)


def sectionize_documents(
    documents: Iterable[str], document_ids: Iterable, disable_progress_bar: bool = False
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
    for document_id, document in tqdm(
        zip(document_ids, documents), total=len(documents), disable=disable_progress_bar
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


def process_documents(
    documents: Iterable[str],
    document_ids: Iterable,
    split_sentences: bool = True,
    filter_len: int = 3,
    disable_progress_bar: bool = False,
) -> pd.DataFrame:
    """
    Main helper function to process documents from the EMR.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param document_type: String denoting the document type to be processed
    :param document_sections: List of sections for a given document type to process
    :param split_sentences: Flag to determine whether to further split sections into sentences
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(df.text.values, df.document_id.values, df.offset.values, filter_len, disable_progress_bar)
    return df


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
    sentence_index = faiss.index_cpu_to_gpu(res, 0, sentence_index, co)
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
        _df = pd.read_parquet(f"{wiki_dir}/{file}", columns=["id", "text"])
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
    contexts = []
    for r in tqdm(df.itertuples(), total=len(df)):
        prompt_id = r.Index
        prompt_indices = processed_wiki_text_data[
            processed_wiki_text_data["document_id"].isin(
                wikipedia_file_data[wikipedia_file_data["prompt_id"] == prompt_id]["id"].values
            )
        ].index.values
        if prompt_indices.shape[0] > 0:
            prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
            prompt_index.add(wiki_data_embeddings[prompt_indices])
            context = ""
            ## Get the top matches
            ss, ii = prompt_index.search(question_embeddings, num_sentences_include)
            for _s, _i in zip(ss[prompt_id], ii[prompt_id]):
                context += processed_wiki_text_data.loc[prompt_indices]["text"].iloc[_i] + " "
        contexts.append(context)
    return contexts


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

        df.reset_index(inplace=True, drop=True)
        if cfg.debug:
            df = df.head()
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

        # wikipedia text data 取得 ("id", "text")
        print("【wikipedia text data 取得】")
        wiki_text_data = get_full_text_data(
            wikipedia_file_data,
            cfg.wiki_dir,
        )
        print(wiki_text_data.head())

        ## Parse documents into sentences
        print("【Parse documents into sentences】")
        processed_wiki_text_data = process_documents(wiki_text_data.text.values, wiki_text_data.id.values)
        print(processed_wiki_text_data.head())

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

        ## Extract contexts from matching pairs
        print("【Extract contexts from matching pairs】")
        contexts = extract_contexts_from_matching_pairs(
            df,
            processed_wiki_text_data,
            wikipedia_file_data,
            wiki_data_embeddings,
            question_embeddings,
            num_sentences_include=cfg.num_sentences_include,
        )
        df["context"] = contexts

        # 保存
        print("【保存】")
        preprocessed_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(preprocessed_path / f"{Path(path).stem}.csv", index=False)

        del wiki_data_embeddings
        del question_embeddings
        del df
        del search_score
        del search_index
        del wikipedia_file_data
        del wiki_text_data
        del processed_wiki_text_data
        _ = gc.collect()


if __name__ == "__main__":
    main()
