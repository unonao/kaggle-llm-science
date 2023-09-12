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
from faiss import read_index, write_index
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

libc = ctypes.CDLL("libc.so.6")
sys.path.append(os.pardir)

import utils


def extract_sections(title: str, text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"#{2,}\s?(.*?)\s?#{2,}")
    sections = []

    matches = list(pattern.finditer(text))
    start_idx = 0

    if len(matches) == 0:
        sections.append((title, text))
        return sections

    for i, match in enumerate(matches):
        if i == 0:
            end_idx = match.start()
            sections.append((title, text[start_idx:end_idx].strip()))

        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = (match.group(1).strip(), text[start_idx:end_idx].strip())
        if section[0] not in ["See also", "References", "Further reading", "External links"]:
            sections.append(section)

        start_idx = end_idx

    # 空のtextの場合は飛ばす
    sections = [section for section in sections if len(section[1].split(" ")) >= 3]
    return sections


def compress_and_split_sections(
    sections: list[tuple[str, str]], max_sentence_length: int, max_sentence_num: int, filter_len=3
) -> list[str]:
    # セクション同士を結合する

    document_sentences = []
    for title, content in sections:
        document = f"{title or 'No Title'}: {content}" + "\n"
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

    # 空のセクションをフィルタリング
    sections = [section for section in document_sentences if len(section) > 0]
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
        sections = []
        ids = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            secs = compress_and_split_sections(
                extract_sections(row["title"], row["text"]), cfg.max_sentence_length, cfg.max_sentence_num
            )
            sections += secs
            ids += [row["id"]] * len(secs)

        sections_df = pd.DataFrame({"id": ids, "section_text": sections})
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
        gc.collect()

        if cfg.debug:
            break


if __name__ == "__main__":
    main()
