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

train_paths = [
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

save_dir = "preprocessed/900_concat"

df = pd.concat([pd.read_csv(path) for path in train_paths])
enwiki_df = pd.read_csv(enwiki_path, index_col=0)

train_df = pd.concat([df, enwiki_df.iloc[1000:]]).reset_index(drop=True)
valid_df = pd.concat([pd.read_csv(train_path), enwiki_df.iloc[:1000]]).reset_index(drop=True)

train_df = train_df[["prompt", "A", "B", "C", "D", "E", "answer"]]
valid_df = valid_df[["prompt", "A", "B", "C", "D", "E", "answer"]]

print(train_df.shape, valid_df.shape)
# trainの場合、80%をfold0, 20%をfold1に分割。valの場合全てfold2に分割
train_df["fold"] = 0
valid_df["fold"] = 2
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for fold, (train_index, val_index) in enumerate(kf.split(train_df)):
    train_df.loc[train_index, "fold"] = 0
    train_df.loc[val_index, "fold"] = 1
    break

train_df.to_csv(os.path.join(save_dir, "train_all.csv"), index=False)
valid_df.to_csv(os.path.join(save_dir, "valid_all.csv"), index=False)
