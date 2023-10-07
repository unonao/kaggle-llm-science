"""
1stの結果を読み込んで、確率値の最大値を予測とする

"""
ignore_index = [
    201,
    205,
    207,
    213,
    217,
    226,
    228,
    239,
    245,
    251,
    254,
    258,
    259,
    260,
    262,
    270,
    278,
    284,
    286,
    296,
    308,
    312,
    314,
    338,
    347,
    359,
    373,
    389,
    401,
    404,
    413,
    417,
    422,
    430,
    438,
    440,
    444,
    451,
    454,
    465,
    467,
    478,
    486,
    491,
    494,
    507,
    508,
    509,
    515,
    530,
    531,
    538,
    550,
    569,
    574,
    586,
    593,
    612,
    623,
    631,
    641,
    646,
    647,
    648,
    656,
    659,
    666,
    667,
    668,
    669,
    670,
    676,
    677,
    689,
    690,
    696,
    698,
    699,
    705,
    716,
    735,
    748,
    753,
    771,
    773,
    779,
    795,
    802,
    803,
    804,
    806,
    819,
    828,
    840,
    841,
    843,
    854,
    856,
    860,
    864,
    874,
    884,
    905,
    914,
    916,
    921,
    928,
    938,
    950,
    957,
    961,
    965,
    971,
    974,
    975,
    979,
    981,
    984,
    995,
    1008,
    1009,
    1011,
    1018,
    1026,
    1028,
    1055,
    1058,
    1061,
    1067,
    1078,
    1083,
    1125,
    1134,
    1143,
    1157,
    1159,
    1165,
    1180,
    1183,
    1185,
    1188,
    1194,
]

import os
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


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


def mistake_idx(true_items, predictions):
    """
    trueとpredのtop1がそれぞれ違うidxを返す
    """
    U = len(predictions)
    mistake_idx = []
    for u in range(U):
        user_preds = predictions[u]
        user_true = true_items[u]
        if user_preds[0] != user_true:
            mistake_idx.append(u)
    return mistake_idx


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

    # data1 = pd.read_csv(cfg.data1_path)
    data2 = pd.read_csv(cfg.data2_path).drop(ignore_index)
    data3 = pd.read_csv(cfg.data3_path)

    # numpy の予測結果を読み込む
    # pred1_list = []
    pred2_list = []
    pred3_list = []
    for dir_path in cfg.pred_dirs:
        # pred1_list.append(np.load(f"{dir_path}/data1_pred.npy"))
        pred2_list.append(np.delete(np.load(f"{dir_path}/data2_pred.npy"), ignore_index, axis=0))
        pred3_list.append(np.load(f"{dir_path}/data3_pred.npy"))

    # 確率値のmaxを取って、予測結果を作成
    # pred1 = np.max(pred1_list, axis=0)
    pred2 = np.max(pred2_list, axis=0)
    pred3 = np.max(pred3_list, axis=0)

    # 確率値を保存
    np.save(output_path / "data2_pred.npy", pred2)

    pred2 = predictions_to_map_output(pred2)
    pred3 = predictions_to_map_output(pred3)
    true2 = data2["answer"].values
    true3 = data3["answer"].values
    map2 = map_k(true2, pred2)
    map3 = map_k(true3, pred3)
    print(f"map2:{map2}, map3:{map3}")

    # top1 の予測が違うidxを抽出
    print("mistake_idx2:", mistake_idx(true2, pred2))
    print("mistake_idx3:", mistake_idx(true3, pred3))


if __name__ == "__main__":
    main()
