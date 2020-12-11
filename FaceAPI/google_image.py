import requests
import re
from bs4 import BeautifulSoup


def google_image_search_by_url(image_url):
    google_image_search_url = 'https://www.google.com/searchbyimage'

    query_params = {'site': 'search', 'sa': 'X', 'image_url': image_url}
    header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    r = requests.get(google_image_search_url, params=query_params, headers=header)
    #print(r.text)
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup.find_all(text=re.compile("Best guess for this image")):
        for child in tag.parent.find_all('a'):
            print(child.string)
            return child.string
