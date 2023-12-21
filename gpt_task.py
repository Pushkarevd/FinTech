import requests
from config import URL, body, promt

from main import ParserGPT


if __name__ == "__main__":
    task = input('Введите задачу оптимизации: ')
    content = promt + task
    body['messages'][0].update({'content': content})

    response = requests.post(URL, json=body)

    task = '\n'.join(response.json()['choices'][0]['message']['content'].split('\n')[:-1])

    print(task)

    instance = ParserGPT(task)

    instance.solve()