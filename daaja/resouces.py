import gzip
import io
import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import List

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class Resouces:
    DIR = Path("/tmp/eda_ja")
    WORDNET_PATH = DIR / Path("wnjpn.db.gz")
    WORDNER_URL = "https://github.com/bond-lab/wnja/releases/download/v1.1/wnjpn.db.gz"
    STOPWORDS_PATH = DIR / Path("stopwords.txt")
    STOPWORDS_URL = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"

    def __init__(self) -> None:
        os.makedirs(self.DIR, exist_ok=True)

        # Download and Load wordnet
        self.download_wordnet()
        connect = sqlite3.connect(self.WORDNET_PATH)
        query = 'SELECT synset, lemma FROM sense, word USING (wordid) WHERE sense.lang="jpn"'
        self.sence_words = pd.read_sql(query, connect)

        #  Download and Load stopwords
        self.download_stopwords()
        with open(self.STOPWORDS_PATH, encoding='utf-8') as f:
            stopwords_txt = f.read()
        self.stopwords = [line.strip() for line in stopwords_txt.split("\n") if len(line.strip()) > 0]

    def download_stopwords(self) -> None:
        if os.path.isfile(self.STOPWORDS_PATH):
            logging.info("The stopword file already exists.")
            return
        res = requests.get(self.STOPWORDS_URL).content.decode("utf-8")
        with open(self.STOPWORDS_PATH, "w", encoding='utf-8') as f:
            f.write(res)

    def download_wordnet(self) -> None:
        if os.path.isfile(self.WORDNET_PATH):
            logging.info("The wordnet file already exists.")
            return

        logging.info("Downloading wordnet.")
        with gzip.open(io.BytesIO(requests.get(self.WORDNER_URL).content), "rb") as f_gz:
            with open(self.WORDNET_PATH, "wb") as f:
                shutil.copyfileobj(f_gz, f)

    def get_synonyms(self, word: str) -> List[str]:
        synsets = self.sence_words.loc[self.sence_words.lemma == word, "synset"]
        synset_words = set(self.sence_words.loc[self.sence_words.synset.isin(synsets), "lemma"])

        if word in synset_words:
            synset_words.remove(word)
        return list(synset_words)

    def is_stopword(self, word: str) -> bool:
        return word in self.stopwords


if __name__ == "__main__":
    resources = Resouces()
    resources.get_synonyms("データ")
