import requests

fh_API_KEY = 'cuvnbk9r01qh55k5uko0cuvnbk9r01qh55k5ukog'
fh_url = f'https://finnhub.io/api/v1/news?category=general&token={fh_API_KEY}'

response = requests.get(fh_url)
news_data = response.json()

