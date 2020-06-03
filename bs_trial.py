import bs4
from urllib.request import urlopen
from PIL import Image
from bs4 import BeautifulSoup as soup
from io import BytesIO
import requests
import numpy as np


def open_page(url):
    uclient = urlopen(my_url)
    page = uclient.read()
    uclient.close()
    page_soup = soup(page, 'lxml')
    return page_soup
search_item = input("Enter search key: ")
search_item = search_item.replace(' ', '+')
my_url = f'https://www.ebay.co.uk/sch/i.html?_sop=12&_sadis=15&_dmd=1&LH_Complete=1&_stpos=LS29NZ&_from=R40&_nkw={search_item}&_sacat=0&_ipg=200'

listing_page = open_page(my_url)
item_list = listing_page.find_all('li', class_='clearfix')
price_list = []
unsold_items = []
for item in item_list:
    img_url = item.find('img', class_='img')['src']
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    img.show()
    try:
        price_item = item.find('span', class_ = 'bidsold').text
        price_list.append(price_item)
    except AttributeError:
        continue
        # price_item = item.find('span', class_ = 'binsold').text
        # unsold_items.append(price_item)
    img_data = item.find_all('div', class_='img')
    item_page_url = img_data[0].find('a')['href']
    item_page_soup = open_page(item_page_url)
    #to gain further info about product: item_page_soup.find('a', class_='ppcvip-db')['href']
    breakpoint()
