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
python exp/001_additional_data.py exp001="000" 
python exp/001_additional_data.py exp001="001"
python exp/001_additional_data.py exp001="002"
python exp/001_additional_data.py exp001="003"
python exp/001_additional_data.py exp001="004"
```

```sh
kaggle datasets create -p llm-science-models --dir-mode zip
kaggle datasets version -p llm-science-models/ -m v1.0.2  --dir-mode zip
```
