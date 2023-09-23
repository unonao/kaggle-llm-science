"""
1stの結果を読み込んで重み付けを行う

"""


import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm.auto import tqdm
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

from scipy.optimize import minimize


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


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c: DictConfig) -> None:
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.exp

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.exp.split('/')[-1]}"
    output_path = Path(f"./output/{exp_name}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(cfg)

    data1 = pd.read_csv(cfg.data1_path)
    data2 = pd.read_csv(cfg.data2_path)
    data3 = pd.read_csv(cfg.data3_path)

    # numpy の予測結果を読み込む
    pred1_list = []
    pred2_list = []
    pred3_list = []
    for dir_path in cfg.pred_dirs:
        pred1_list.append(np.load(f"{dir_path}/data1_pred.npy"))
        pred2_list.append(np.load(f"{dir_path}/data2_pred.npy"))
        pred3_list.append(np.load(f"{dir_path}/data3_pred.npy"))

    n = len(pred1_list)
    initial_weights = [1.0 / n for _ in range(n)]
    bounds = [(0, 1) for _ in range(n)]

    def objective(weights):
        pred1 = np.zeros_like(pred1_list[0])
        for i, w in enumerate(weights):
            pred1 += w * pred1_list[i]
        pred1 = predictions_to_map_output(pred1)
        true1 = data1["answer"].values
        map1 = map_k(true1, pred1)
        return -map1

    res = minimize(objective, initial_weights, bounds=bounds, method="Nelder-Mead")
    weights = res.x / res.x.sum()
    print("weights:", weights)

    # 重みを元にdata2, data3の予測結果を作成
    pred2 = np.zeros_like(pred2_list[0])
    pred3 = np.zeros_like(pred3_list[0])
    for i, w in enumerate(weights):
        pred2 += w * pred2_list[i]
        pred3 += w * pred3_list[i]
    pred2 = predictions_to_map_output(pred2)
    pred3 = predictions_to_map_output(pred3)
    true2 = data2["answer"].values
    true3 = data3["answer"].values
    map2 = map_k(true2, pred2)
    map3 = map_k(true3, pred3)
    print(f"map2:{map2}, map3:{map3}")


if __name__ == "__main__":
    main()
