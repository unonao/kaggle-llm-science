"""
1stの結果を読み込んでlightgbmで学習する

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


@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c: DictConfig) -> None:
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.exp

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.exp.split('/')[-1]}"
    output_path = Path(f"./output/{exp_name}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(cfg)

    unuse_col = ["df_index", "option_index", "option", "label"]

    data_dir_dict = OmegaConf.to_container(cfg.data_dir_dict)
    train_df = None
    valid_df = None
    for data_name, data_dir in data_dir_dict.items():
        t_df = pd.read_csv(os.path.join(data_dir, "data1_2nd.csv"))
        v_df = pd.read_csv(os.path.join(data_dir, "data2_2nd.csv"))

        # 最初に学習に使わないカラムはセット
        if train_df is None:
            train_df = t_df[unuse_col]
            valid_df = v_df[unuse_col]

        # 学習に使わないカラムは drop して、他のカラムには data_name を名前に結合
        t_df = t_df.drop(unuse_col, axis=1)
        t_df = t_df.add_prefix(f"{data_name}_")
        v_df = v_df.drop(unuse_col, axis=1)
        v_df = v_df.add_prefix(f"{data_name}_")

        # 学習データと検証データに結合
        train_df = pd.concat([train_df, t_df], axis=1)
        valid_df = pd.concat([valid_df, v_df], axis=1)

    print(train_df.shape, valid_df.shape)
    print(train_df.head())

    # LightGBM で学習
    use_cols = [col for col in train_df.columns if col not in unuse_col]
    print(use_cols)
    lgb_train = lgb.Dataset(train_df[use_cols], train_df["label"])
    lgb_eval = lgb.Dataset(valid_df[use_cols], valid_df["label"], reference=lgb_train)
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
    # 予測
    pred = bst.predict(valid_df[use_cols])
    valid_df["pred"] = pred
    # df_index ごとに、predの順でOptionのリストを作成
    valid_df = valid_df.sort_values(["df_index", "pred"], ascending=[True, False])
    pred_df = valid_df.groupby("df_index").agg({"option": list}).reset_index()
    pred_df["gt"] = valid_df[valid_df["label"] == 1].option.values
    # map@3 を計算
    ## 全体
    data2_map3 = map_k(pred_df["gt"].values, pred_df["option"].values, K=3)
    print(f"data2_map3: {data2_map3}")
    train_csv_map3 = map_k(pred_df["gt"].head(200).values, pred_df["option"].head(200).values, K=3)
    print(f"train_csv_map3: {train_csv_map3}")


if __name__ == "__main__":
    main()
