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
    AutoModel,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
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
        labels = [feature.pop(label_name) for feature in features]
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
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
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

    print(cfg)

    utils.seed_everything(cfg.seed)

    wandb.init(
        project="kaggle-llm-science-infer",
        name=exp_name,
        mode="online" if cfg.debug is False else "disabled",
        config=OmegaConf.to_container(cfg),
    )

    # os.makedirs(output_path, exist_ok=True)

    base_dir = Path(cfg.data1_path).parent
    df_valid3 = pd.read_csv(cfg.data3_path).reset_index(drop=True)
    df_valid4 = pd.read_csv(base_dir / "val_500_enhanced.csv").reset_index(drop=True)

    if cfg.debug:
        df_valid3 = df_valid3.head(10)
        df_valid4 = df_valid4.head(10)
    print(f"valid3:{df_valid3.shape} valid4:{df_valid4.shape}")

    def preprocess_df(df, mode="train"):
        max_length = cfg.max_length if mode == "train" else cfg.max_length_valid  # 推論時はtokenを長く取る
        df["prompt_with_context"] = (
            df["context"].fillna("no context").apply(lambda x: " ".join(x.split()[:max_length]))
            + f"... {cfg.sep_token} "
            + df["prompt"].fillna("")
        )
        df["prompt_with_context"] = df["prompt_with_context"].apply(clean_text)

        # 空を埋める
        options = ["A", "B", "C", "D", "E"]
        for option in options:
            df[option] = df[option].fillna("")
        return df

    df_valid3 = preprocess_df(df_valid3, mode="valid")
    df_valid4 = preprocess_df(df_valid4, mode="valid")

    dataset_valid3 = Dataset.from_pandas(df_valid3)
    dataset_valid4 = Dataset.from_pandas(df_valid4)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    option_to_index = {option: idx for idx, option in enumerate("ABCDE")}
    index_to_option = {v: k for k, v in option_to_index.items()}

    def preprocess(example):
        first_sentence = [example["prompt_with_context"]] * 5
        second_sentences = [example[option] for option in "ABCDE"]
        tokenized_example = tokenizer(first_sentence, second_sentences, truncation=False)
        tokenized_example["label"] = option_to_index[example["answer"]]

        return tokenized_example

    tokenized_dataset_valid3 = dataset_valid3.map(
        preprocess, batched=False, remove_columns=["prompt_with_context", "prompt", "A", "B", "C", "D", "E", "answer"]
    )
    tokenized_dataset_valid4 = dataset_valid4.map(
        preprocess, batched=False, remove_columns=["prompt_with_context", "prompt", "A", "B", "C", "D", "E", "answer"]
    )

    # https://www.kaggle.com/code/philippsinger/h2ogpt-perplexity-ranking
    def precision_at_k(r, k):
        """Precision at k"""
        assert k <= len(r)
        assert k != 0
        return sum(int(x) for x in r[:k]) / k

    def map_k(true_items, predictions, K=3):
        """Score is mean average precision at 3"""
        U = len(predictions)
        map_at_k = 0.0
        for u in range(U):
            user_preds = predictions[u]
            user_true = true_items[u]
            user_results = [1 if item == user_true else 0 for item in user_preds]
            for k in range(min(len(user_preds), K)):
                map_at_k += precision_at_k(user_results, k + 1) * user_results[k]
        return map_at_k / U

    def predictions_to_map_output(predictions):
        sorted_answer_indices = np.argsort(-predictions)  # Sortting indices in descending order
        top_answer_indices = sorted_answer_indices[:, :]  # Taking the first three indices for each row
        top_answers = np.vectorize(index_to_option.get)(
            top_answer_indices
        )  # Transforming indices to options - i.e., 0 --> A
        return np.apply_along_axis(lambda row: " ".join(row), 1, top_answers)

    model = AutoModelForMultipleChoice.from_pretrained(cfg.model_path)
    args = TrainingArguments(output_dir="output/tmp", per_device_eval_batch_size=1)
    trainer = Trainer(
        model=model, args=args, tokenizer=tokenizer, data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer)
    )

    with utils.timer("valid"):
        # valid を確認
        valid3_pred = trainer.predict(tokenized_dataset_valid3).predictions
        valid4_pred = trainer.predict(tokenized_dataset_valid4).predictions
        # torch softmaxをかける
        valid3_pred = torch.softmax(torch.tensor(valid3_pred), dim=1).numpy()
        valid4_pred = torch.softmax(torch.tensor(valid4_pred), dim=1).numpy()

        result_dict = {
            "train_csv_map@3": map_k(df_valid3["answer"].to_numpy(), predictions_to_map_output(valid3_pred)),
            "valid_csv_map@3": map_k(df_valid4["answer"].to_numpy(), predictions_to_map_output(valid4_pred)),
        }
        print(result_dict)
        wandb.log(result_dict)

        # 予測結果をnumpyで保存
        output_path.mkdir(exist_ok=True, parents=True)
        np.save(output_path / "data3_pred.npy", valid3_pred)
        np.save(output_path / "data4_pred.npy", valid4_pred)


if __name__ == "__main__":
    main()