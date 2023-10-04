"""
2つのoptionのうちどちらが正解かを予測するモデルを作成する。
context + prompt + optionA, optionB を入力として、
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    AutoConfig,
)

from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

import wandb

sys.path.append(os.pardir)

import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def clean_text(text):
    text = text.replace('"', "")
    text = text.replace("“", "")
    text = text.replace("”", "")
    return text


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c: DictConfig) -> None:
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.exp

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.exp.split('/')[-1]}"
    output_path = Path(f"./output/{exp_name}")
    cfg.training_args.output_dir = str(output_path)

    print(cfg)

    utils.seed_everything(cfg.seed)

    wandb.init(
        project="kaggle-llm-science-binary",
        name=exp_name,
        mode="online" if cfg.debug is False else "disabled",
        config=OmegaConf.to_container(cfg),
    )

    # os.makedirs(output_path, exist_ok=True)

    option_to_index = {option: idx for idx, option in enumerate("ABCDE")}
    index_to_option = {v: k for k, v in option_to_index.items()}

    # train
    df_train = pd.concat([pd.read_csv(path) for path in cfg.data0_paths]).reset_index(drop=True)
    soft_label = np.concatenate(
        [np.load(Path(cfg.soft_label_dir) / (Path(path).stem + ".npy")) for path in cfg.data0_paths], axis=0
    )
    df_train["first_option"] = np.argsort(soft_label)[:, -1]
    df_train["first_option"] = df_train["first_option"].map(index_to_option)
    df_train["second_option"] = np.argsort(soft_label)[:, -2]
    df_train["second_option"] = df_train["second_option"].map(index_to_option)
    df_train["answer_location"] = "other"
    df_train.loc[df_train["first_option"] == df_train["answer"], "answer_location"] = "first"
    df_train.loc[df_train["second_option"] == df_train["answer"], "answer_location"] = "second"

    # valid
    df_valid = pd.read_csv(cfg.data2_path).reset_index(drop=True)
    soft_label = np.load(Path(cfg.soft_label_dir) / (Path(cfg.data2_path).stem + ".npy"))
    df_valid["first_option"] = np.argsort(soft_label)[:, -1]
    df_valid["first_option"] = df_valid["first_option"].map(index_to_option)
    df_valid["second_option"] = np.argsort(soft_label)[:, -2]
    df_valid["second_option"] = df_valid["second_option"].map(index_to_option)
    df_valid["answer_location"] = "other"
    df_valid.loc[df_valid["first_option"] == df_valid["answer"], "answer_location"] = "first"
    df_valid.loc[df_valid["second_option"] == df_valid["answer"], "answer_location"] = "second"

    if cfg.debug:
        df_train = df_train.head(10)
        df_valid = df_valid.head(10)

    print(f"train:{df_train.shape}, valid:{df_valid.shape}")

    # 最大正解率（first, secondに答えが含まれている確率）
    max_prob = (df_valid["answer_location"] != "other").mean()
    # 元々の正解率
    original_prob = (df_valid["answer"] == df_valid["first_option"]).mean()
    print()
    print(f"max_prob: {max_prob}, original_prob: {original_prob}")
    print()

    def preprocess_df(df, mode="train"):
        max_length = cfg.max_length if mode == "train" else cfg.max_length_valid  # 推論時はtokenを長く取る
        # 空を埋める
        options = ["A", "B", "C", "D", "E"]
        for option in options:
            df[option] = df[option].fillna("")

        # first_option が先に来るケースと、second_option が先に来るケースの２つを作る
        df_first = df.copy()
        df_first["text"] = (
            df_first["context"].apply(lambda x: " ".join(x.split()[:max_length]))
            + f"... [SEP]  "
            + df_first["prompt"]
            + " [SEP] Wrong: "
            + df_first["second_option"]
            + " [SEP] Correct: "
            + df_first["first_option"]  # ここが正解なら [0, 1]
        )
        df_first["label"] = df_first["answer_location"].map(
            {"first": [0.0, 1.0], "second": [1.0, 0.0], "other": [0.0, 0.0]}
        )

        df_second = df.copy()
        df_second["text"] = (
            df_second["context"].apply(lambda x: " ".join(x.split()[:max_length]))
            + f"... [SEP]  "
            + df_second["prompt"]
            + " [SEP] Wrong: "
            + df_second["first_option"]
            + " [SEP] Correct: "
            + df_second["second_option"]  # ここが正解なら [0, 1]
        )
        df_second["label"] = df_second["answer_location"].map(
            {"first": [1.0, 0.0], "second": [0, 1.0], "other": [0.0, 0.0]}
        )

        df = pd.concat([df_first, df_second]).reset_index(drop=True)
        df["text"] = df["text"].apply(clean_text)

        return df

    df_train = preprocess_df(df_train)
    df_valid = preprocess_df(df_valid, mode="valid")

    dataset_train = Dataset.from_pandas(df_train)
    dataset_valid = Dataset.from_pandas(df_valid)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def preprocess(example):
        tokenized_example = tokenizer(example["text"], truncation=False)
        tokenized_example["label"] = example["label"]  # クラス0: 後半に正解が含まれる確率, クラス1: 前半に正解が含まれる確率
        return tokenized_example

    tokenized_dataset_train = dataset_train.map(
        preprocess, remove_columns=["prompt", "A", "B", "C", "D", "E", "answer"]
    )
    tokenized_dataset_valid = dataset_valid.map(
        preprocess, remove_columns=["prompt", "A", "B", "C", "D", "E", "answer"]
    )

    def compute_metrics(eval_preds):
        """
        正解率を計算する
        """
        logits, labels = eval_preds
        labels = np.array(labels)
        pred_ids = np.argmax(logits, axis=1)
        latter = labels[:, 0]
        latter[latter < 0.6] = 0
        latter[latter > 0.6] = 1
        former = labels[:, 1]
        former[former < 0.6] = 0
        former[former > 0.6] = 1

        latter = latter.astype(int)
        former = former.astype(int)

        # 後半に正解が含まれると予測
        latter_correct = (pred_ids == 1) & (latter == 1)
        # 前半に正解が含まれると予測
        former_correct = (pred_ids == 0) & (former == 1)
        correct = latter_correct | former_correct
        return {"accuracy": correct.mean()}

    ## Training
    training_args = TrainingArguments(**OmegaConf.to_container(cfg)["training_args"])

    config = AutoConfig.from_pretrained(cfg.model_name, num_labels=2, problem_type="multi_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_valid,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        compute_metrics=compute_metrics,
    )

    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    with utils.timer("valid"):
        valid_pred = trainer.predict(tokenized_dataset_valid).predictions

        # 最大正解率（first, secondに答えが含まれている確率）
        max_prob = (df_valid["answer_location"] != "other").mean()
        # 元々の正解率
        original_prob = (df_valid["answer"] == df_valid["first_option"]).mean()
        # 予測による正解率
        original_len = len(df_valid) // 2
        pred = (valid_pred[:original_len] + valid_pred[original_len:, [1, 0]]) / 2
        predict_prob = (
            np.argmax(pred, axis=1)  # 前半分は後半がfirst, 前半がsecond
            == df_valid.head(original_len)["answer_location"].map({"first": 1, "second": 0, "other": -1})
        ).mean()

        result_dict = {
            "max_prob": max_prob,
            "original_prob": original_prob,
            "predict_prob": predict_prob,
        }
        print(result_dict)
        wandb.log(result_dict)

        # 予測結果をnumpyで保存
        np.save(output_path / "pred.npy", valid_pred)


if __name__ == "__main__":
    main()
