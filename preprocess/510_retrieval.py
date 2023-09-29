"""
filter_4_2.parquetなどのデータを読み込んで、embeddingを計算し、faissのindexを作成する。
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm.auto import tqdm

import re
import faiss
from faiss import read_index, write_index

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import ctypes
import os
import gc

libc = ctypes.CDLL("libc.so.6")
sys.path.append(os.pardir)

import utils


def retrieval(
    df: pd.DataFrame,
    index_path: str,
    model: SentenceTransformer,
    top_k: int = 3,
    batch_size: int = 32,
) -> pd.DataFrame:
    sentence_index = read_index(index_path)  # index 読み込み
    res = faiss.StandardGpuResources()  # use a single GPU
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    sentence_index = faiss.index_cpu_to_gpu(res, 0, sentence_index, co)
    prompt_embeddings = model.encode(
        df.prompt_answer_stem.values,
        batch_size=batch_size,
        device="cuda",
        show_progress_bar=True,
        # convert_to_tensor=True,
        normalize_embeddings=True,
    )
    prompt_embeddings = prompt_embeddings.astype(np.float32)
    # prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
    search_score, search_index = sentence_index.search(prompt_embeddings, top_k)
    res.noTempMemory()
    del res
    del sentence_index
    del prompt_embeddings
    _ = gc.collect()
    libc.malloc_trim(0)
    return search_score, search_index


def extract_context(
    row,
    context_series: pd.Series,
) -> str:
    search_index = row["search_index"]
    context = []
    for i in search_index:
        context.append(context_series.iloc[i])
    return " ".join(context)


def clean_prompt_answer(i):
    if "What is" in i.prompt:
        answer = ""
        answer = answer + i.prompt[7:-1] + " " + "is that" + " " + i.A
        answer = answer + i.prompt[7:-1] + " " + "is that" + " " + i.B
        answer = answer + i.prompt[7:-1] + " " + "is that" + " " + i.C
        answer = answer + i.prompt[7:-1] + " " + "is that" + " " + i.D
        answer = answer + i.prompt[7:-1] + " " + "is that" + " " + i.E
    elif "What was" in i.prompt:
        answer = ""
        answer = answer + i.prompt[8:-1] + " " + "was that" + " " + i.A
        answer = answer + i.prompt[8:-1] + " " + "was that" + " " + i.B
        answer = answer + i.prompt[8:-1] + " " + "was that" + " " + i.C
        answer = answer + i.prompt[8:-1] + " " + "was that" + " " + i.D
        answer = answer + i.prompt[8:-1] + " " + "was that" + " " + i.E
    elif "What are" in i.prompt:
        answer = ""
        answer = answer + i.prompt[8:-1] + " " + "are that" + " " + i.A
        answer = answer + i.prompt[8:-1] + " " + "are that" + " " + i.B
        answer = answer + i.prompt[8:-1] + " " + "are that" + " " + i.C
        answer = answer + i.prompt[8:-1] + " " + "are that" + " " + i.D
        answer = answer + i.prompt[8:-1] + " " + "are that" + " " + i.E
    elif "What were" in i.prompt:
        answer = ""
        answer = answer + i.prompt[9:-1] + " " + "were that" + " " + i.A
        answer = answer + i.prompt[9:-1] + " " + "were that" + " " + i.B
        answer = answer + i.prompt[9:-1] + " " + "were that" + " " + i.C
        answer = answer + i.prompt[9:-1] + " " + "were that" + " " + i.D
        answer = answer + i.prompt[9:-1] + " " + "were that" + " " + i.E
    elif "Which of the following statements" in i.prompt:
        answer = ""
        answer = i.prompt.replace("Which of the following statements ", "This ") + " ".join([i.A, i.B, i.C, i.D, i.E])
        answer = answer.replace("?", ". ")
    elif "Which of the following is" in i.prompt:
        answer = ""
        answer = i.prompt.replace("Which of the following is ", "This is ") + " ".join([i.A, i.B, i.C, i.D, i.E])
        answer = answer.replace("?", ". ")
    elif "What did " in i.prompt:
        answer = ""
        answer = i.prompt.replace("What did ", "") + " ".join([i.A, i.B, i.C, i.D, i.E])
        answer = answer.replace("?", ". ")
    elif "How do " in i.prompt:
        answer = ""
        answer = i.prompt.replace("How do ", "") + " ".join([i.A, i.B, i.C, i.D, i.E])
        answer = answer.replace("?", ". ")
    elif "How did " in i.prompt:
        answer = ""
        answer = i.prompt.replace("How did ", "") + " ".join([i.A, i.B, i.C, i.D, i.E])
        answer = answer.replace("?", ". ")
    else:
        pattern = r"Which (\w+ \w+ \w+|\w+ \w+|\w+) is"
        replacement = r"This \1 is"
        result = re.sub(pattern, replacement, i.prompt)
        if result != i.prompt:
            answer = result + " ".join([i.A, i.B, i.C, i.D, i.E])
            answer = answer.replace("?", ". ")
        else:
            answer = i.prompt + " " + " ".join([i.A, i.B, i.C, i.D, i.E])
    return answer


# hydraで設定を読み込む
@hydra.main(version_base=None, config_path="../yamls", config_name="config")
def main(c: DictConfig) -> None:
    OmegaConf.resolve(c)  # debugやseedを解決
    cfg = c.preprocess

    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).stem}/{runtime_choices.preprocess.split('/')[-1]}"
    preprocessed_path = Path(f"./preprocessed/{exp_name}")
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    print(cfg)
    print("preprocessed_path:", preprocessed_path)

    # モデル読み込み
    model = SentenceTransformer(cfg.sim_model, device="cuda")
    model.max_seq_length = cfg.max_length
    model = model.half()

    for path in cfg.data_paths:
        # データ読み込み
        df = pd.read_csv(path)
        df[["A", "B", "C", "D", "E"]] = df[["A", "B", "C", "D", "E"]].fillna("")

        df.reset_index(inplace=True, drop=True)
        if cfg.debug:
            df = df.head(15)
        print(f"{path}:{df.shape}")

        df["prompt_answer_stem"] = df.apply(clean_prompt_answer, axis=1)

        search_score, search_index = retrieval(
            df,
            cfg.index_path,
            model,
            top_k=cfg.top_k,
            batch_size=cfg.batch_size,
        )
        df["search_index"] = search_index.tolist()
        wiki_df = pd.read_parquet(cfg.filter_wiki_path)
        df["context"] = df.apply(lambda x: extract_context(x, wiki_df["context"]), axis=1)
        df = df.drop(columns=["search_index", "prompt_answer_stem"])
        df.to_csv(preprocessed_path / f"{Path(path).stem}.csv", index=False)


if __name__ == "__main__":
    main()
