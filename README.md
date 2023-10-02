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


python preprocess/311_embedding_b.py preprocess=311/003


python preprocess/325_fts_db.py preprocess=325/000
```

tm

### その他

1st

```sh
python preprocess/330_retrieve_a.py preprocess=330/000
python preprocess/330_retrieve_a.py preprocess=330/001

python preprocess/331_retrieve_b.py preprocess=331/000 
python exp/300_1st.py exp=300/000
python preprocess/331_retrieve_b.py preprocess=331/001
python exp/300_1st.py exp=300/001

python preprocess/331_retrieve_b.py preprocess=331/b_bge_10_5_4

python exp/350_1st_infer.py exp=350/a_gte_10_3_2
```

2nd

```sh
python preprocess/350_2nd_option.py preprocess=350/000 # base
python preprocess/340_2nd_data.py preprocess=340/000 
python exp/400_2nd.py exp=400/000


python preprocess/340_2nd_data.py preprocess=340/a_gte_10_3_2 
python preprocess/400_base_data.py
python exp/400_2nd.py exp=400/100

# blend
python exp/500_blend.py exp=500/000
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
kaggle datasets version -p dataset/llm-science-index/ -m v1.6.0  --dir-mode zip

kaggle datasets init -p dataset/llm-science-filter-index
kaggle datasets create -p dataset/llm-science-filter-index --dir-mode zip
kaggle datasets version -p dataset/llm-science-filter-index/ -m v1.2.0  --dir-mode zip

kaggle datasets create -p llm-science-wikipedia --dir-mode zip
kaggle datasets version -p llm-science-wikipedia  -m v1.0.0 

kaggle datasets init -p llm-science-lgb
kaggle datasets create -p llm-science-lgb --dir-mode zip
kaggle datasets version -p dataset/llm-science-lgb  -m v1.4.0

```

```sh

python preprocess/335_retrieve_a_improve.py preprocess=335/a_gte_10_3_2
python preprocess/335_retrieve_a_improve.py preprocess=335/a_gte_10_4_3
python preprocess/336_retrieve_b_improve.py preprocess=336/b_bge_10_4_3
python preprocess/336_retrieve_b_improve.py preprocess=336/b_multi_10_4_3
python preprocess/336_retrieve_b_improve.py preprocess=336/b_bge_base_10_4_3
python preprocess/336_retrieve_b_improve.py preprocess=336/b_minilm_10_4_3

python preprocess/500_index.py preprocess=500/parse_multi
python preprocess/500_index.py preprocess=500/nparse_bge
python preprocess/500_index.py preprocess=500/nparse_multi
python preprocess/500_index.py preprocess=500/parse_bge
python preprocess/510_retrieval.py preprocess=510/parse_bge
python preprocess/510_retrieval.py preprocess=510/parse_multi
python preprocess/510_retrieval.py preprocess=510/nparse_bge
python preprocess/510_retrieval.py preprocess=510/nparse_multi
python exp/350_1st_infer.py exp=350/parse_bge
python exp/350_1st_infer.py exp=350/parse_multi
python exp/350_1st_infer.py exp=350/nparse_bge
python exp/350_1st_infer.py exp=350/nparse_multi

# lightgbmのためにinferのdata0のheadを削除して実行
python exp/350_1st_infer.py exp=350/new_a_gte_10_3_2
python exp/350_1st_infer.py exp=350/new_a_gte_10_4_3
python exp/350_1st_infer.py exp=350/new_b_bge_10_4_3
python exp/350_1st_infer.py exp=350/new_b_multi_10_4_3
python exp/350_1st_infer.py exp=350/new_b_bge_base_10_4_3
python exp/350_1st_infer.py exp=350/new_b_minilm_10_4_3

python exp/600_max.py exp=500/107

# ローカル：maxでの調整、lightgbmの活用検討、シングルモデルの改善方法検討
# submit: 510系のsubをまずはシングルで→アンサンブルで


python preprocess/500_index.py preprocess=500/parse_e5
python preprocess/500_index.py preprocess=500/parse_gte
python preprocess/500_index.py preprocess=500/nparse_e5
python preprocess/510_retrieval.py preprocess=510/parse_e5
python preprocess/510_retrieval.py preprocess=510/nparse_e5
python preprocess/510_retrieval.py preprocess=510/parse_gte
python exp/350_1st_infer.py exp=350/parse_e5
python exp/350_1st_infer.py exp=350/nparse_e5
python exp/350_1st_infer.py exp=350/parse_gte

# まだアンサンブルを試していない
python preprocess/500_index.py preprocess=500/nparse_gte
python preprocess/510_retrieval.py preprocess=510/nparse_gte
python exp/350_1st_infer.py exp=350/nparse_gte
python preprocess/500_index.py preprocess=500/parse_bge_base
python preprocess/500_index.py preprocess=500/nparse_bge_base
python preprocess/500_index.py preprocess=500/parse_gte_base
python preprocess/500_index.py preprocess=500/nparse_gte_base
python preprocess/510_retrieval.py preprocess=510/parse_bge_base
python preprocess/510_retrieval.py preprocess=510/parse_gte_base
python preprocess/510_retrieval.py preprocess=510/nparse_bge_base
python preprocess/510_retrieval.py preprocess=510/nparse_gte_base
python exp/350_1st_infer.py exp=350/parse_bge_base
python exp/350_1st_infer.py exp=350/parse_gte_base
python exp/350_1st_infer.py exp=350/nparse_bge_base
python exp/350_1st_infer.py exp=350/nparse_gte_base


# これから
python preprocess/500_index.py preprocess=500/parse_e5_base
python preprocess/500_index.py preprocess=500/nparse_e5_base
python preprocess/500_index.py preprocess=500/parse_multi_base
python preprocess/500_index.py preprocess=500/nparse_multi_base
python preprocess/510_retrieval.py preprocess=510/parse_e5_base
python preprocess/510_retrieval.py preprocess=510/nparse_e5_base
python preprocess/510_retrieval.py preprocess=510/parse_multi_base
python preprocess/510_retrieval.py preprocess=510/nparse_multi_base
python exp/350_1st_infer.py exp=350/parse_e5_base
python exp/350_1st_infer.py exp=350/nparse_e5_base
python exp/350_1st_infer.py exp=350/parse_multi_base
python exp/350_1st_infer.py exp=350/nparse_multi_base

```

```sh
python exp/360_infer_label.py exp=360/000 
python exp/302_1st_soft.py exp=302/000
python exp/303_1st_soft_v2.py exp=302/000 
```
