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

prompt

```sh
python preprocess/901_concat_no_wikibase.py 
```

wikipedia

```sh
# please download enwiki-20230701-pages-articles-multistream.xml.bz2  
cd wikiextractor
pip install .
cd ..
python -m wikiextractor.WikiExtractor input/enwiki-20230701-pages-articles-multistream.xml.bz2  --processes 8  --json -b 1G -o input/enwiki-20230701
python preprocess/300_wiki_data_a.py 
## yet
python preprocess/310_embedding_a.py preprocess=310/000 
python preprocess/320_doc_index.py preprocess=320/000


# download https://dumps.wikimedia.org/other/cirrussearch/current/enwiki-20230911-cirrussearch-content.json.gz
cd wikiextractor
pip install .
cd ..
python -m wikiextractor.cirrus-extract input/enwiki-20230911-cirrussearch-content.json.gz  -b 1G -o input/enwiki-20230911-cirrus
python preprocess/301_wiki_data_b.py
python preprocess/311_embedding_b.py preprocess=311/000
python preprocess/320_doc_index.py preprocess=320/001

python preprocess/311_embedding_b.py preprocess=311/001
python preprocess/320_doc_index.py preprocess=320/002
```

### その他

1st

```sh
python preprocess/330_retrieve_a.py preprocess=330/000
python preprocess/330_retrieve_a.py preprocess=330/001

python preprocess/331_retrieve_b.py preprocess=331/000 
python exp/300_1st.py exp=300/000
python preprocess/331_retrieve_b.py preprocess=331/001
python exp/300_1st.py exp=300/001

python exp/350_1st_infer.py exp=350/000
python exp/350_1st_infer.py exp=350/001
python exp/350_1st_infer.py exp=350/002
python exp/350_1st_infer.py exp=350/003

python exp/350_1st_infer.py exp=350/b_multi_5_4_3
python exp/350_1st_infer.py exp=350/a_gte_5_4_3
python exp/350_1st_infer.py exp=350/b_bge_5_4_3

python preprocess/331_retrieve_b.py preprocess=331/b_multi_10_4_3
python preprocess/331_retrieve_b.py preprocess=331/b_multi_10_5_4
python preprocess/330_retrieve_a.py preprocess=330/a_gte_10_3_2
python preprocess/330_retrieve_a.py preprocess=330/a_gte_10_4_3
python preprocess/331_retrieve_b.py preprocess=331/b_bge_10_3_2 
python preprocess/331_retrieve_b.py preprocess=331/b_bge_10_4_3 
python preprocess/331_retrieve_b.py preprocess=331/b_bge_10_5_4
```

2nd

```sh
python preprocess/350_2nd_option.py preprocess=350/000 # base
# each model
python preprocess/340_2nd_data.py preprocess=340/000 
python exp/400_2nd.py exp=400/000

# blend
python exp/500_blend.py exp=500/000
```

旧

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

kaggle dataset

```sh
cd dataset
kaggle datasets init -p llm-science-wikipedia-data-a
zip -r  llm-science-wikipedia-data-a/data.zip input/llm-science-wikipedia-data-a
kaggle datasets create -p llm-science-wikipedia-data-a --dir-mode zip

kaggle datasets init -p llm-science-wikipedia-data-b
zip -r  llm-science-wikipedia-data-b/data.zip input/llm-science-wikipedia-data-b
kaggle datasets create -p llm-science-wikipedia-data-b --dir-mode zip

kaggle datasets create -p llm-science-models --dir-mode zip
kaggle datasets version -p llm-science-models/ -m v1.8.0  --dir-mode zip

kaggle datasets init -p llm-science-index
kaggle datasets create -p llm-science-index --dir-mode zip
kaggle datasets version -p llm-science-index/ -m v1.5.0  --dir-mode zip

kaggle datasets create -p llm-science-wikipedia --dir-mode zip
kaggle datasets version -p llm-science-wikipedia  -m v1.0.0 

kaggle datasets init -p llm-science-lgb
kaggle datasets create -p llm-science-lgb --dir-mode zip
kaggle datasets version -p llm-science-lgb  -m v1.1.0

```
