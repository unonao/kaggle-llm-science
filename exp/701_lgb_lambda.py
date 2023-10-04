"""
1stの結果を読み込んで、確率値の最大値を予測とする

"""


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
import lightgbm as lgb


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


def make_datasets(df: pd.DataFrame, pred_list: list[np.ndarray], num_from: int = 0) -> pd.DataFrame:
    """
    optionごとに行に分割して、それぞれの行に対してpredを結合する
    """
    records = []
    for di, row in df[num_from:].iterrows():
        for oi, option in enumerate("ABCDE"):
            record = {
                "df_index": di,
                "option_index": oi,
                "option": option,
                "label": option == row["answer"],
            }
            for pi in range(len(pred_list)):
                record[f"pred{pi}"] = pred_list[pi][di, oi]
            records.append(record)
    return pd.DataFrame(records)


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
    data2 = pd.read_csv(cfg.data2_path)
    data3 = pd.read_csv(cfg.data3_path)

    # numpy の予測結果を読み込む
    # pred1_list = []
    pred2_list = []
    pred3_list = []
    for dir_path in cfg.pred_dirs:
        # pred1_list.append(np.load(f"{dir_path}/data1_pred.npy"))
        pred2_list.append(np.load(f"{dir_path}/data2_pred.npy"))
        pred3_list.append(np.load(f"{dir_path}/data3_pred.npy"))

    dataset2 = make_datasets(data2, pred2_list, num_from=200)
    dataset3 = make_datasets(data3, pred3_list)

    print(dataset2.shape)

    unuse_col = ["df_index", "option_index", "option", "label"]
    print(dataset2.head())
    use_cols = [col for col in dataset2.columns if col not in unuse_col]
    print(use_cols)
    lgb_train = lgb.Dataset(dataset2[use_cols], dataset2["label"], group=[5] * 1000)
    lgb_eval = lgb.Dataset(dataset3[use_cols], dataset3["label"], group=[5] * 200, reference=lgb_train)
    bst = lgb.train(
        OmegaConf.to_container(cfg.lgbm.params),
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=cfg.lgbm.num_boost_round,
        callbacks=[lgb.log_evaluation(cfg.lgbm.verbose_eval)],
    )

    print(f"best_itelation: {bst.best_iteration}, train: {bst.best_score['train']}, valid: {bst.best_score['valid']}")

    # importanceを保存
    fig, ax = plt.subplots(figsize=(10, 20))
    ax = lgb.plot_importance(bst, importance_type="gain", ax=ax, max_num_features=100)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "importance.png"))
    # モデルを保存
    bst.save_model(os.path.join(output_path, "model.txt"))

    # 保存したモデルを読み込む
    bst = lgb.Booster(model_file=os.path.join(output_path, "model.txt"))
    # train
    pred = bst.predict(dataset2[use_cols])
    dataset2["pred"] = pred
    dataset2 = dataset2.sort_values(["df_index", "pred"], ascending=[True, False])
    pred_df = dataset2.groupby("df_index").agg({"option": list}).reset_index()
    pred_df["gt"] = dataset2[dataset2["label"] == 1].option.values
    data2_map3 = map_k(pred_df["gt"].values, pred_df["option"].values, K=3)
    print(f"data2_map3: {data2_map3}")

    # valid
    pred = bst.predict(dataset3[use_cols])
    dataset3["pred"] = pred
    dataset3 = dataset3.sort_values(["df_index", "pred"], ascending=[True, False])
    pred_df = dataset3.groupby("df_index").agg({"option": list}).reset_index()
    pred_df["gt"] = dataset3[dataset3["label"] == 1].option.values
    data3_map3 = map_k(pred_df["gt"].values, pred_df["option"].values, K=3)
    print(f"data3_map3: {data3_map3}")


if __name__ == "__main__":
    main()
