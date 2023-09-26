"""
Wikipedia Plaintext データセットの各ファイルについて、テキストをセクションごとに分割し、sentence-transformerで埋め込みを作成して保存する
"""
from __future__ import annotations

import ctypes
import gc
import glob
import os
import re
import sys
from pathlib import Path

import blingfire as bf
import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pandarallel import pandarallel
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

pandarallel.initialize(progress_bar=True)
libc = ctypes.CDLL("libc.so.6")
sys.path.append(os.pardir)

import utils

'''
def extract_chunk_by_sliding_window(text_list: list[str], window_size: int, sliding_size: int) -> list[str]:
    """
    text のリストをsliding windowで結合する。window_size個のtextが含まれるまで結合し、sliding_size個ずつずらして結合する。
    """
    chunks = []
    for i in range(0, len(text_list), sliding_size):
        chunk = " ".join(text_list[i : i + window_size])
        chunks.append(chunk)
    return chunks
'''


def compress_and_split_sections(
    document: str, title: str, max_sentence_length: int, max_sentence_num: int, filter_len=3
) -> list[str]:
    # documentを分割し、適切な長さになるように結合する

    document_sentences = []
    section_sentences = []
    try:
        _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
        for o in sentence_offsets:
            if o[1] - o[0] > filter_len:
                sentence = document[o[0] : o[1]]
                section_sentences.append(sentence)
    except:
        section_sentences = [document]

    buffer = ""

    for text in section_sentences:
        if len((buffer + text).split(" ")) <= max_sentence_length:
            buffer += text + "\n"
        else:
            document_sentences.append(buffer.strip())
            buffer = text + "\n"

    if buffer:
        document_sentences.append(buffer.strip())

    # 空のセクションをフィルタリング + titleを追加
    sections = [title + " : " + section for section in document_sentences if len(section) > 0]
    return sections[:max_sentence_num]


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
    model.half()

    # データ読み込み
    for path in glob.glob(f"{cfg.wiki_dir}/*.parquet"):
        # 存在するファイルは飛ばす
        if (preprocessed_path / f"{Path(path).stem}.parquet").exists():
            continue
        # index は除外
        if "index" in path or "all" in path:
            continue

        print("path:", path)
        df = pd.read_parquet(path)
        if cfg.debug:
            df = df.head(100)
        print("df.shape:", df.shape)

        # セクションごとに分割し compress_sections で、もとのidと一緒に新たなdfを作成
        """
        sections = []
        ids = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            secs = compress_and_split_sections(
                row["text"], row["title"], cfg.max_sentence_length, cfg.max_sentence_num
            )
            sections += secs
            ids += [row["id"]] * len(secs)
        """
        # 上記をapplyで書くと早い
        df["sections"] = df.parallel_apply(
            lambda row: compress_and_split_sections(
                row["text"], row["title"], cfg.max_sentence_length, cfg.max_sentence_num
            ),
            axis=1,
        )

        sections_df = (
            pd.DataFrame(
                {
                    "id": df["id"],
                    "section_text": df["sections"],
                }
            )
            .explode(["section_text"])
            .reset_index(drop=True)
        )
        print("sections_df.shape:", sections_df.shape)

        # 埋め込みを作成
        section_embeddings = model.encode(
            sections_df.section_text.tolist(),
            batch_size=cfg.batch_size,
            device="cuda",
            show_progress_bar=True,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        # 型をfloat16に変換
        section_embeddings = section_embeddings.astype(np.float16)
        print("section_embeddings.shape:", section_embeddings.shape)

        # section_dfと埋め込みを保存
        os.makedirs(preprocessed_path, exist_ok=True)
        sections_df.to_parquet(preprocessed_path / f"{Path(path).stem}.parquet")
        np.save(preprocessed_path / f"{Path(path).stem}.npy", section_embeddings)
        # メモリ解放
        del df
        del sections_df
        del section_embeddings
        _ = gc.collect()
        libc.malloc_trim(0)

        if cfg.debug:
            break


if __name__ == "__main__":
    main()
