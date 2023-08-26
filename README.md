# Kaggle テンプレート

## How to use

### 準備

- WEBHOOK_URL 環境変数にセットしておくこと

### 実行

```sh
docker compose biuld
docker compose run --rm kaggle bash # bash に入る
docker compose up # jupyter lab 起動
```

```sh
python exp/001_additional_data.py exp=001/000 debug=true
python exp/001_additional_data.py exp=001/001
python exp/001_additional_data.py exp=001/002
python exp/001_additional_data.py exp=001/003
python exp/001_additional_data.py exp=001/004
python exp/002_additional_datas.py exp=002/000
python exp/002_additional_datas.py exp=002/001
python exp/002_additional_datas.py exp=002/002
python preprocess/000_base.py preprocess=000/000
python exp/003_retrieval.py exp=003/000 debug=True
python exp/004_retrieval_truncate.py exp=004/000
python exp/004_retrieval_truncate.py exp=004/001
```

```sh
kaggle datasets create -p llm-science-models --dir-mode zip
kaggle datasets version -p llm-science-models/ -m v1.1.1  --dir-mode zip
```
