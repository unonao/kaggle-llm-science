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
python exp/001_additional_data.py exp001="000" debug=True 
python exp/001_additional_data.py exp001="001"
python exp/001_additional_data.py exp001="002"
python exp/001_additional_data.py exp001="003"
```
