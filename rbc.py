from typing import Union, Tuple, List, Dict, Any

import requests
import pandas as pd
from tqdm import tqdm

from bs4 import BeautifulSoup
from datetime import datetime, timedelta


class RBC:
    header: dict[str, str] = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'
    }

    def __init__(self):
        self._session = requests.Session()

    @staticmethod
    def form_url(dt: int):
        return f'https://quote.rbc.ru/v5/ajax/get-news-feed/project/quote/lastDate/{dt}/limit/20'

    def get_chunk(self, url: str) -> Union[tuple[None, None], tuple[list[dict[str, Union[str, Any]]], Any]]:
        response = self._session.get(url=url, headers=self.header)
        data = response.json()['items']
        parsed_chunk = [self.parse_html(article['html']) for article in data]
        if not parsed_chunk:
            print(response.status_code)
            return None, None
        return parsed_chunk, parsed_chunk[-1]['timestamp']

    def start_parsing(self):
        dt_pointer = int(datetime.now().timestamp())

        result = []

        for _ in tqdm(range(2000)):
            url = self.form_url(dt_pointer)
            chunk, dt_pointer = self.get_chunk(url)

            result.extend(chunk)

            dt_pointer = int(dt_pointer)

        df = pd.DataFrame(result)
        print(df)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.drop_duplicates(inplace=True)
        df.to_csv('./data/rbc.csv', index=False)

    def get_tags(self, url: str) -> str:
        response = self._session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        tags = soup.find_all('a', {'class': 'article__tags__item'})
        if len(tags) > 0:
            return tags[0].text

    def parse_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')

        parsed_html = {
            'url': soup.a['href'],
            'title': soup.find('span', {'class': 'news-feed__item__title'}).text.strip(),
            'timestamp': soup.a['data-modif'],
            'tag': self.get_tags(soup.a['href'])
        }
        return parsed_html


if __name__ == '__main__':
    instance = RBC()
    instance.start_parsing()
