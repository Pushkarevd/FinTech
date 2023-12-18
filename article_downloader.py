import requests
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from goose3 import Goose
from numpy import array_split


class Loader:
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'
    }

    def __init__(self, filename: str):
        self.filename = filename
        self._df = pd.read_csv(self.filename)
        self._extractor = Goose()
        self._target_filename = f'ready_data.csv'

    def load_one_article(self, row: pd.Series) -> str:
        url = row.url
        response = requests.get(url, headers=self.header)
        article = self._extractor.extract(raw_html=response.text)
        return article.cleaned_text

    def add_text_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df['article_text'] = df.apply(self.load_one_article, axis=1)
        return df

    def download_articles_text(self, n_workers=10) -> None:
        chunks = array_split(self._df, n_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as worker:
            futures = []
            resulted_chunks = []
            for chunk in chunks:
                futures.append(worker.submit(self.add_text_column, chunk))

            for future in futures:
                resulted_chunks.append(future.result())

        pd.concat(resulted_chunks).to_csv(self._target_filename, index=False)


if __name__ == '__main__':
    instance = Loader(filename='./data/rbc.csv')
    instance.download_articles_text()
