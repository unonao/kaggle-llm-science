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
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from tqdm.auto import tqdm
import wandb
import logging

sys.path.append(os.pardir)

import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

logging.disable(logging.WARNING)


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
        project="kaggle-llm-science-pipeline",
        name=exp_name,
        mode="online" if cfg.debug is False else "disabled",
        config=OmegaConf.to_container(cfg),
    )

    # os.makedirs(output_path, exist_ok=True)

    df_valid2 = pd.read_csv(cfg.data2_path).reset_index(drop=True)
    df_valid3 = pd.read_csv(cfg.data3_path).reset_index(drop=True)
    if cfg.debug:
        df_valid2 = df_valid2.head(10)
        df_valid3 = df_valid3.head(10)
    print(f"valid2:{df_valid2.shape}, valid3:{df_valid3.shape}")

    def preprocess_df(df):
        cols = ["prompt", "A", "B", "C", "D", "E"]
        # clean
        for col in cols:
            df[col] = df[col].apply(clean_text)
        # 空を埋める
        options = ["A", "B", "C", "D", "E"]
        for option in options:
            df[option] = df[option].fillna("")
        return df

    df_valid2 = preprocess_df(df_valid2)
    df_valid3 = preprocess_df(df_valid3)

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

    option_to_index = {option: idx for idx, option in enumerate("ABCDE")}
    index_to_option = {v: k for k, v in option_to_index.items()}

    def predictions_to_map_output(predictions):
        sorted_answer_indices = np.argsort(-predictions)  # Sortting indices in descending order
        top_answer_indices = sorted_answer_indices[:, :]  # Taking the first three indices for each row
        top_answers = np.vectorize(index_to_option.get)(
            top_answer_indices
        )  # Transforming indices to options - i.e., 0 --> A
        return np.apply_along_axis(lambda row: " ".join(row), 1, top_answers)

    model_name_or_path = "TheBloke/Mistral-7B-v0.1-GPTQ"  # "/kaggle/input/mistral-7b-gptq/Mistral-7B-v0.1-GPTQ/"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", trust_remote_code=False, revision="main"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    MAX_CONTEXT = 1700

    # from https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag
    def get_tokens(question, context, options, tokenizer):
        system_prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_prefix}"
        instruction = "Your task is to analyze the question and answer below. Here A,B,C,D,E options are given choose the correct one after Analysing. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be relevant."
        prompt_suffix = "".join([f"{letter}: {options[letter]}\n\n" for letter in "ABCDE"])
        input_prefix = f"Context: {context[:MAX_CONTEXT]}\n\nQuestion: {question}\n\nOptions: {prompt_suffix}\n\nProposed answer: "

        prompt_prefix = system_prefix.format(instruction=instruction, input_prefix=input_prefix)
        return tokenizer(prompt_prefix, return_tensors="pt").to(f"cuda:{model.device.index}")

    answers_token_id = tokenizer.encode("A B C D E")[1:]

    def get_result(df):
        df_valid = df.reset_index()[["index", "prompt", "A", "B", "C", "D", "E", "context", "answer"]]
        predictions = []
        for index in tqdm(range(df_valid.shape[0])):
            columns = df_valid.iloc[index].values
            question = columns[1]
            options = {"A": columns[2], "B": columns[3], "C": columns[4], "D": columns[5], "E": columns[6]}
            context1 = columns[7]
            inputs1 = get_tokens(question, context1, options, tokenizer)
            with torch.no_grad():
                outputs1 = model.generate(
                    input_ids=inputs1["input_ids"],
                    attention_mask=inputs1["attention_mask"],
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                first_token_probs1 = outputs1.scores[0][0]
                option_scores1 = first_token_probs1[answers_token_id].float().cpu().numpy()  # ABCDE
                probability1 = torch.softmax(torch.tensor(option_scores1), dim=-1).numpy()
                predictions.append(probability1)
        return np.array(predictions)

    with utils.timer("valid"):
        # valid を確認
        valid2_pred = get_result(df_valid2)
        valid3_pred = get_result(df_valid3)

        result_dict = {
            "data2_map@3": map_k(df_valid2["answer"].to_numpy(), predictions_to_map_output(valid2_pred)),
            "train_csv_map@3": map_k(df_valid3["answer"].to_numpy(), predictions_to_map_output(valid3_pred)),
        }
        print(result_dict)
        wandb.log(result_dict)

        # 予測結果をnumpyで保存
        output_path.mkdir(exist_ok=True, parents=True)
        np.save(output_path / "data2_pred.npy", valid2_pred)
        np.save(output_path / "data3_pred.npy", valid3_pred)


if __name__ == "__main__":
    main()
