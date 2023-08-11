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
import wandb
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

sys.path.append(os.pardir)

import utils

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


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c : DictConfig) -> None:    
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.exp001

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.exp001}"
    output_path = Path(f"./output/{exp_name}")
    cfg.training_args.output_dir = str(output_path)

    print(cfg)    

    utils.seed_everything(cfg.seed)

    wandb.init(
        project="kaggle-llm-science-holdout",
        name=exp_name,
        mode="online" if cfg.debug is False else "disabled",
        config=OmegaConf.to_container(cfg),
    )

    # os.makedirs(output_path, exist_ok=True)

    df_valid = pd.read_csv(cfg.comp_data_path + "/train.csv")
    df_valid = df_valid.drop(columns="id")
    df_train = pd.read_csv(cfg.additional_data_path + "/6000_train_examples.csv").head(cfg.use_train_num)
    df_train.reset_index(inplace=True, drop=True)
    if cfg.debug:
        df_train = df_train.head()
        df_valid = df_valid.head()
    print(f"train:{df_train.shape}, valid:{df_valid.shape}")

    dataset_train = Dataset.from_pandas(df_train)
    dataset_valid = Dataset.from_pandas(df_valid)


    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    option_to_index = {option: idx for idx, option in enumerate("ABCDE")}
    index_to_option = {v: k for k, v in option_to_index.items()}
    def preprocess(example):
        first_sentence = [example["prompt"]] * 5
        second_sentences = [example[option] for option in "ABCDE"]
        tokenized_example = tokenizer(first_sentence, second_sentences, truncation=False)
        tokenized_example["label"] = option_to_index[example["answer"]]

        return tokenized_example
    
    tokenized_dataset_train = dataset_train.map(preprocess, remove_columns=["prompt", "A", "B", "C", "D", "E", "answer"])
    tokenized_dataset_valid = dataset_valid.map(preprocess, remove_columns=["prompt", "A", "B", "C", "D", "E", "answer"])

    def map_k(y_true, y_pred, k=3):
        ap_at_3 = 0.0
        for i in range(len(y_true)):
            if y_true[i] in y_pred[i][:k]:
                ap_at_3 += 1
        return ap_at_3 / len(y_true)


    def predictions_to_map_output(predictions):
        sorted_answer_indices = np.argsort(-predictions)  # Sortting indices in descending order
        top_answer_indices = sorted_answer_indices[:, :]  # Taking the first three indices for each row
        top_answers = np.vectorize(index_to_option.get)(
            top_answer_indices
        )  # Transforming indices to options - i.e., 0 --> A
        return np.apply_along_axis(lambda row: " ".join(row), 1, top_answers)


    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        y_pred = predictions_to_map_output(logits)
        y_true = [index_to_option[label] for label in labels]
        return {cfg.training_args.metric_for_best_model: map_k(y_true, y_pred)}

    ## Training
    training_args = TrainingArguments(
        **OmegaConf.to_container(cfg)["training_args"]
    )

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

    trainer.train()

    with utils.timer("valid"):
        # valid を確認
        valid_pred = trainer.predict(tokenized_dataset_valid).predictions
        valid_pred_letters = predictions_to_map_output(valid_pred)
        valid_label = df_valid["answer"].to_numpy()
        valid_map3 = map_k(valid_label, valid_pred_letters)
        result_dict = {"best_map@3":valid_map3 }
        print(result_dict)
        wandb.log(result_dict)

if __name__ == "__main__":
    main()