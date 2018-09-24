# coding:utf-8
import urllib
import pandas as pd
import os
import sys
import io
import uuid
import traceback
from tqdm import tqdm_notebook as tqdm
from bs4 import BeautifulSoup
from PIL import Image


MY_EMAIL = "kmintei1998@yahoo.co.jp"
POKEMON_NUMBER = 808


class Fetcher:
    def __init__(self, ua=''):
        self.ua = ua

    def fetch(self, url, timeout=3):
        req = urllib.request.Request(url, headers={'User-Agent': self.ua})
        try:
            p = urllib.request.urlopen(url, timeout=timeout)
        except:
            sys.stderr.write('Error in fetching {}\n'.format(url))
            sys.stderr.write(traceback.format_exc())
            return None
        return p


class ImageScraiper(Fetcher):

    def get_image(self, search_word, image_number_range=(1, 50), save_path="", engine="yahoo", html_parser="lxml"):
        if engine == "yahoo":
            search_word = urllib.parse.urlencode({"p": search_word})
            for page in tqdm(range(image_number_range[0], image_number_range[1], 20), desc="image pages :", leave=False):
                search_url = "https://search.yahoo.co.jp/image/search?{}&ei=UTF-8&b={}&ktot=7".format(
                    search_word, page)
                structured_html = self.fetch(search_url)
                bs4_html = BeautifulSoup(structured_html, html_parser)
                image_src_list = bs4_html.find_all("div", class_="SeR")
                for image_src in tqdm(image_src_list, desc="image #", leave=False):
                    img = image_src.img.get("src")
                    img = self.fetch(img)
                    if img is None:
                        continue
                    else:
                        img = img.read()
                    img_bin = io.BytesIO(img)
                    img = Image.open(img_bin)
                    path = os.path.join(save_path, str(uuid.uuid4()) + ".png")
                    img.save(path + str(uuid.uuid4()) + ".png", format="PNG")


def get_pokemon_many_image(debug=False, pokemon_data=None, path="./pokemon_img", image_number_range=(1, 100), pokemon_number=POKEMON_NUMBER, my_email=MY_EMAIL):
    imagescraiper = ImageScraiper(my_email)
    if debug:
        pokemon_number = 3
        image_number_range = (1, 3)
    for _number in tqdm(range(1, pokemon_number), desc="pokemon #"):
        save_path = os.path.join(path, str(_number))
        pokemon_name = pokemon_data.loc[_number - 1, "Name"]
        imagescraiper.get_image(
            pokemon_name, image_number_range=image_number_range, save_path=save_path)
