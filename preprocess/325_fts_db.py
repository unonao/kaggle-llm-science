import os
import re
import sqlite3
import sys
from pathlib import Path

import blingfire as bf
import hydra
import nltk
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from omegaconf import DictConfig, OmegaConf
from pandarallel import pandarallel
from tqdm.auto import tqdm

pandarallel.initialize(progress_bar=True)


# hydraで設定を読み込む
@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c: DictConfig) -> None:
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.preprocess
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.preprocess.split('/')[-1]}"
    preprocessed_path = Path(f"./preprocessed/{exp_name}")
    # フォルダ作成
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    db_path = preprocessed_path / "fts.db"

    def extract_chunk_by_sliding_window(text_list: list[str], window_size: int, sliding_size: int) -> list[str]:
        """
        text のリストをsliding windowで結合する。window_size個のtextが含まれるまで結合し、sliding_size個ずつずらして結合する。
        """
        chunks = []
        for i in range(0, len(text_list), sliding_size):
            chunk = " ".join(text_list[i : i + window_size])
            chunks.append(chunk)
        return chunks

    def split_sentences(text):
        document = text.replace("\n", " ")
        _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
        section_sentences = []
        for o in sentence_offsets:
            section_sentences.append(document[o[0] : o[1]])
        chunks = extract_chunk_by_sliding_window(section_sentences, cfg.window_size, cfg.sliding_size)
        return chunks

    # db 作成
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    cur = db.cursor()
    cur.execute('create virtual table imdb using fts5(text, tokenize="porter unicode61");')  # UNINDEXED

    parquet_paths = [path for path in list(Path(cfg.parquet_dir).glob("*.parquet")) if "all" not in path.name]
    parquet_paths.sort()
    # 一つずつ読み込んで、dbに書き込む
    for i, path in enumerate(parquet_paths):
        # tqdmでpathを出力
        print(str(path))

        if cfg.debug:
            if i > 1:
                break

        df = pd.read_parquet(path)
        if cfg.debug:
            df = df.head(100)

        df["text"] = df["text"].parallel_apply(split_sentences)
        df = df.explode(["text"]).reset_index(drop=True)
        df["text"] = df["title"] + " > " + df["text"]

        cur.executemany(
            "insert into imdb (text) values (?);",
            df[["text"]].to_records(index=False),
        )
        db.commit()

    # dbを閉じる
    db.close()


if __name__ == "__main__":
    main()
