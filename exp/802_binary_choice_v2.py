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
    AutoModelForMultipleChoice,
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


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        # labels = [feature.pop(label_name) for feature in features]
        soft_labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(soft_labels, dtype=torch.float32)
        return batch


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

    print(df_train.head())

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
            df[option] = df[option].fillna("No text.")

        df["first_option_text"] = df.apply(lambda row: row[row["first_option"]], axis=1)
        df["second_option_text"] = df.apply(lambda row: row[row["second_option"]], axis=1)

        df["prompt_with_context"] = (
            df["context"].apply(lambda x: " ".join(x.split()[:max_length])) + f"... [SEP] " + df["prompt"]
        )
        df["prompt_with_context"] = df["prompt_with_context"].apply(clean_text)

        return df

    df_train_processed = preprocess_df(df_train)
    df_valid_processed = preprocess_df(df_valid, mode="valid")

    dataset_train = Dataset.from_pandas(df_train_processed)
    dataset_valid = Dataset.from_pandas(df_valid_processed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def preprocess(example):
        sentences = [
            example["prompt_with_context"]
            + f" [SEP] "
            + example["second_option_text"]
            + f" [SEP] "
            + example["first_option_text"],
            example["prompt_with_context"]
            + f" [SEP] "
            + example["first_option_text"]
            + f" [SEP] "
            + example["second_option_text"],
        ]
        tokenized_example = tokenizer(sentences, truncation=False)
        if example["answer_location"] == "first":
            tokenized_example["label"] = [1.0, 0.0]
        elif example["answer_location"] == "second":
            tokenized_example["label"] = [0.0, 1.0]
        else:
            tokenized_example["label"] = [0.5, 0.5]
        return tokenized_example

    tokenized_dataset_train = dataset_train.map(
        preprocess, remove_columns=["prompt_with_context", "prompt", "A", "B", "C", "D", "E", "answer"]
    )
    tokenized_dataset_valid = dataset_valid.map(
        preprocess, remove_columns=["prompt_with_context", "prompt", "A", "B", "C", "D", "E", "answer"]
    )

    def compute_metrics(eval_preds):
        """
        正解率を計算する
        """
        logits, labels = eval_preds
        logits = torch.softmax(torch.tensor(logits), dim=1).numpy()
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
        latter_correct = (pred_ids == 0) & (latter == 1)
        # 前半に正解が含まれると予測
        former_correct = (pred_ids == 1) & (former == 1)
        correct = latter_correct | former_correct
        return {"accuracy": correct.mean()}

    ## Training
    training_args = TrainingArguments(**OmegaConf.to_container(cfg)["training_args"])

    model = AutoModelForMultipleChoice.from_pretrained(cfg.model_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_valid,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        compute_metrics=compute_metrics,
    )

    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    with utils.timer("valid"):
        valid_pred = trainer.predict(tokenized_dataset_valid).predictions
        valid_pred = torch.softmax(torch.tensor(valid_pred), dim=1).numpy()

        # 最大正解率（first, secondに答えが含まれている確率）
        max_prob = (df_valid_processed["answer_location"] != "other").mean()
        # 元々の正解率
        original_prob = (df_valid_processed["answer"] == df_valid_processed["first_option"]).mean()
        # 予測による正解率
        predict_prob = (
            np.argmax(valid_pred, axis=1)  # 前半分は後半がfirst, 前半がsecond
            == df_valid_processed["answer_location"].map({"first": 0, "second": 1, "other": -1})
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
