"""
input/enwiki-20230911-cirrus 以下のファイルを全て読み取って結合し、
先頭１文字目から file 名として a.parquet,...z.parquet, number.parquet, other.parquet を決定
全体に保存後、それぞれのファイルに分割して保存する
"""

import gc
import glob
import os
import re
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

debug = False
input_dir = Path("input/enwiki-20230911-cirrus/AA")
save_dir = "input/llm-science-wikipedia-cirrus"


def get_first_char(text):
    """先頭１文字目を返す"""
    return text[0].lower()


def map_first_chat2filename(first_char):
    """先頭１文字目からファイル名を決定"""
    if re.match(r"[a-z]", first_char):
        return first_char + ".parquet"
    elif re.match(r"\d", first_char):
        return "number.parquet"
    else:
        return "other.parquet"


def main():
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame()

    paths = glob.glob(f"{input_dir}/*")
    if debug:
        paths = paths[:1]

    # ファイルの各行は {"id": "", "revid": "", "url": "", "title": "", "text": "..."} のjson形式
    print("read files")
    for path in tqdm(paths, total=len(paths)):
        tmp_df = pd.read_json(path, lines=True)
        # text が空のものは除外
        tmp_df = tmp_df[tmp_df.text != ""]
        df = pd.concat([df, tmp_df])
        del tmp_df
        gc.collect()
    # 先頭１文字目からファイル名を決定
    print("get file name")
    df["file"] = df["title"].map(get_first_char).map(map_first_chat2filename)

    # revid, url を drop
    df = df.drop(["revid", "url"], axis=1)
    gc.collect()

    # "id" をstringに
    df["id"] = df["id"].astype(str)
    print(df.head())

    # 全体を保存
    print("reset index")
    df = df.reset_index(drop=True)
    gc.collect()

    # file ごとに分割して保存
    print("save each file")
    for file in tqdm(df.file.unique()):
        df[df.file == file].drop("file", axis=1).reset_index(drop=True).to_parquet(f"{save_dir}/{file}")
        gc.collect()

    print("save all")
    # text を drop
    df = df.drop("text", axis=1)
    df.to_parquet(f"{save_dir}/wiki-all-index.parquet")
    gc.collect()


if __name__ == "__main__":
    main()
