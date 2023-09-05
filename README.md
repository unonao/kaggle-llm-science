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
python exp/003_retrieval.py exp=003/000 
python exp/004_retrieval_truncate.py exp=004/000
python exp/004_retrieval_truncate.py exp=004/001
python exp/005_retrieval.py exp=005/000 
python exp/005_retrieval.py exp=005/001
python exp/006_add_valid.py exp=006/000 
python exp/006_add_valid.py exp=006/000 
python exp/007_validation.py exp=007/000 
python exp/007_validation.py exp=007/001


python preprocess/000_base.py preprocess=000/000
python preprocess/001.py preprocess=001/000
python preprocess/002_gpu.py preprocess=002/000
python preprocess/100_embedding.py preprocess=100/000 # TODO a.npy はdebugになっている
```

```sh
kaggle datasets create -p llm-science-models --dir-mode zip
kaggle datasets version -p llm-science-models/ -m v1.2.0  --dir-mode zip
```
