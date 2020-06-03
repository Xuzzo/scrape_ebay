import requests
from io import BytesIO
from PIL import Image
from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
import os


def open_page(url):
    uclient = urlopen(my_url)
    page = uclient.read()
    uclient.close()
    page_soup = soup(page, 'lxml')
    return page_soup

search_item = input("Enter search key: ")
search_item = search_item.replace(' ', '+')
DATA_PATH = os.path.join('/Users/mmfp/Desktop', 'pokemon_cards_ds')
pok_folder = os.path.join(DATA_PATH, search_item)
true_folder = os.path.join(pok_folder, 'true')
false_folder = os.path.join(pok_folder, 'false')

if not os.path.exists(pok_folder):
    os.mkdir(pok_folder)
    os.mkdir(true_folder)
    os.mkdir(false_folder)

my_url = f'https://www.ebay.co.uk/sch/i.html?_sop=12&_sadis=15&_dmd=1&LH_Complete=1&_stpos=LS29NZ&_from=R40&_nkw={search_item}&_sacat=0&_ipg=200'

listing_page = open_page(my_url)
item_list = listing_page.find_all('li', class_='clearfix')
counter_y = 0
counter_n = 0
for item in item_list:
    try:
        img_url = item.find('img', class_='img')['imgurl']
    except KeyError:
        img_url = item.find('img', class_='img')['src']
    except TypeError:
        continue
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    img.show()
    label = input("Correct card? ")
    if label == 'y':
        counter_y += 1
        file_path = os.path.join(true_folder, search_item+str(counter_y)+'.png')
        img.save(file_path)
    elif label == 'n':
        counter_n += 1
        file_path = os.path.join(false_folder, search_item+str(counter_n)+'.png')
        img.save(file_path)