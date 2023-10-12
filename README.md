# Kaggle - LLM Science Exam 11th solution

## How to use

- WEBHOOK_URL: To be set in the environment variable.

### Docker

```sh
docker compose build
docker compose run --rm kaggle bash # bash に入る
docker compose up # jupyter lab 起動
```

### Making Dataset

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

Examples

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
