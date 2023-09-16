# Kaggle テンプレート

## How to use

### 準備

- WEBHOOK_URL 環境変数にセットしておくこと

### 実行

```sh
docker compose build
docker compose run --rm kaggle bash # bash に入る
docker compose up # jupyter lab 起動
```

### データセット

wikipedia

```sh
# please download enwiki-20230701-pages-articles-multistream.xml.bz2  
cd wikiextractor
pip install .
cd ..
python -m wikiextractor.WikiExtractor input/enwiki-20230701-pages-articles-multistream.xml.bz2  --processes 8  --json -b 1G -o input/enwiki-20230701
python preprocess/300_wiki_data_a.py

# download https://dumps.wikimedia.org/other/cirrussearch/current/enwiki-20230911-cirrussearch-content.json.gz
cd wikiextractor
pip install .
cd ..
python -m wikiextractor.cirrus-extract input/enwiki-20230911-cirrussearch-content.json.gz  -b 1G -o input/enwiki-20230911-cirrus
python preprocess/301_wiki_data_b.py
preprocess/311_embedding_b.py preprocess=311/000 
```

prompt

```sh
python preprocess/900_concat.py # 公開データの結合、fold分割
```

### その他

```sh
python exp/007_validation.py exp=007/006

# bullt 対処前のwikidump
python preprocess/200_wiki.py 
python preprocess/210_embedding.py  preprocess=210/000 debug=True
python preprocess/220_doc_index.py preprocess=220/000 
python preprocess/231_retrieve.py preprocess=230/000  # 前処理追加したので注意
python exp/200_new.py exp=200/000 
python exp/200_new.py exp=200/001
```

```sh
kaggle datasets create -p llm-science-models --dir-mode zip
kaggle datasets version -p llm-science-models/ -m v1.5.0  --dir-mode zip

kaggle datasets init -p llm-science-index
kaggle datasets create -p llm-science-index --dir-mode zip
kaggle datasets version -p llm-science-index/ -m v1.２.0  --dir-mode zip

kaggle datasets create -p llm-science-wikipedia --dir-mode zip
kaggle datasets version -p llm-science-wikipedia  -m v1.0.0 
```
