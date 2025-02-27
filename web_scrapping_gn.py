from googlenews import GoogleNews
import json
import requests
from bs4 import BeautifulSoup
import time

gn = GoogleNews()
countries = ['India', 'UK', 'US', 'Australia', 'Canada', 'China', 'France', 'Germany', 'Italy', 'Japan', 'Russia', 'South Korea']
all_news = []
for country in countries:
    top = gn.geo_headlines(country)
    entries = top['entries']
    count = 0  
    for entry in entries:
        count = count + 1  
        #print (entry["title"])
        all_news.append({'number':str(count),'Country': country,'Title': str(entry["title"])})
    time.sleep(0.5)