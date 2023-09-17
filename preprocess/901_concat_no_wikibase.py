"""
,prompt,context,A,B,C,D,E,answer,source
というカラムのCSVファイルを読み込んで、wikipedia 以外が元となるデータを排除する
source=9 (EduQG) 
source=10,11,12 (sinQ)
"""


import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

save_dir = "preprocessed/901_concat"
os.makedirs(save_dir, exist_ok=True)

non_wiki_base_paths = [
    "input/eduqg-dataset-llm-science-exam-format-34k/eduqg_llm_formatted.csv",
    "input/llm-science-exam-sciq-dataset/test_sciq.csv",
    "input/llm-science-exam-sciq-dataset/train_sciq.csv",
    "input/llm-science-exam-sciq-dataset/valid_sciq.csv",
    "input/40k-data-with-context-v2/MMLU_17k_with_context2.csv",
    "input/40k-data-with-context-v2/OpenBook_with_context2.csv",
    "input/40k-data-with-context-v2/ScienceQA_with_context2.csv",
]
non_wiki_base_df = pd.concat([pd.read_csv(path) for path in non_wiki_base_paths])


# wikipedia がベースとなるデータ
wiki_base_paths = [
    "input/additional-train-data-for-llm-science-exam/6000_train_examples.csv",
    "input/additional-train-data-for-llm-science-exam/extra_train_set.csv",
    "input/sci-or-not-sci-hypthesis-testing-pack/6000_all_categories_questions.csv",
    "input/15k-high-quality-examples/15k_gpt3.5-turbo.csv",
    "input/15k-high-quality-examples/5900_examples.csv",
    "input/llm-science-3k-data/test.csv",
    "input/wikipedia-stem-1k/stem_1k_v1.csv",
]

train_path = "input/kaggle-llm-science-exam/train.csv"
enwiki_path = "input/sci-or-not-sci-hypthesis-testing-pack/6000_wiki_en_sci_questions.csv"


enwiki_df = pd.read_csv(enwiki_path, index_col=0)
wiki_base_df = pd.concat([pd.read_csv(path) for path in wiki_base_paths] + [enwiki_df.iloc[1000:]]).reset_index(
    drop=True
)
df2 = pd.concat([pd.read_csv(train_path), enwiki_df.iloc[:1000]]).reset_index(drop=True)

# 75%をfold0, 25%をfold1に分割。valの場合全てfold2に分割
wiki_base_df["fold"] = 0
kf = KFold(n_splits=4, shuffle=True, random_state=0)
for fold, (train_index, val_index) in enumerate(kf.split(wiki_base_df)):
    wiki_base_df.loc[train_index, "fold"] = 0
    wiki_base_df.loc[val_index, "fold"] = 1
    break

# データを３種作成（data0には wikipedia 以外がベースのデータを結合）
df0 = pd.concat([wiki_base_df[wiki_base_df["fold"] == 0], non_wiki_base_df])
df1 = wiki_base_df[wiki_base_df["fold"] == 1]

# 必要なカラムを残す。option(A,B,C,D,E) が欠損していることがあるので空で埋める
use_cols = ["prompt", "A", "B", "C", "D", "E", "answer"]
df0 = df0[use_cols].fillna("")
df1 = df1[use_cols].fillna("")
df2 = df2[use_cols].fillna("")

print(df0.shape, df1.shape, df2.shape)

# 保存
df0.to_csv(os.path.join(save_dir, "data0.csv"), index=False)
df1.to_csv(os.path.join(save_dir, "data1.csv"), index=False)
df2.to_csv(os.path.join(save_dir, "data2.csv"), index=False)
