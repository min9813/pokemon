# coding: utf-8

import urllib
import pandas as pd
import cv2
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm
from bs4 import BeautifulSoup
from PIL import Image

POKEMON_NUMBER = 808

TEST_URL = "./pokemon_img/daisukiclab/original/1.png"


def get_pokemon_image(path):
    if ("." in path[1:]) or (path[-2] == "/") or (path[-1].isdigit()) or (path[-1] == "/"):
        sys.exit(
            "end of path directory must be parent directory of each pokemon number directory")
    for num in tqdm(range(1, POKEMON_NUMBER)):
        save_path = path + "/" + str(num)
        check_and_make_dir(save_path)
        pokemon_number = "{:0>3}".format(num)
        url_pokemon = "https://www.pokemon.jp/zukan/detail/{}.html".format(
            pokemon_number)
        html = urllib.request.urlopen(url_pokemon).read()
        soup = BeautifulSoup(html, "lxml")
        img_url = "https://www.pokemon.jp" + \
            soup.find("div", class_="profile-phto").img.get("src")
        r = urllib.request.urlopen(img_url).read()
        with open(save_path + "/daisuki_org.png", "wb") as file:
            file.write(r)
        url_pokemon = "https://img.yakkun.com/poke/sm/n{}.gif".format(num)
        img = urllib.request.urlopen(url_pokemon).read()
        with open(save_path + "/yakkun_org.png", "wb") as f:
            f.write(img)


def check_and_make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_pokemon_data(url="http://blog.game-de.com/pm-sm/sm-allstats/", file_name="pokemon_data.csv", encoding="utf-8"):
    url_pokemon_table = url
    print("request to url:{} ...".format(url))
    html = urllib.request.urlopen(url_pokemon_table).read()
    soup = BeautifulSoup(html, "lxml")
    pokemon_table = soup.find("table").tbody.find_all("tr")
    delete_word = ["小", "大", "特大"]
    print("finish request, start data formatting...")
    while len(pokemon_table) > 802:
        for pokemon in pokemon_table:
            if "-" in pokemon.find("td").text:
                pokemon_table.remove(pokemon)
            for word in delete_word:
                if word in pokemon.find("td", class_="b").text:
                    pokemon_table.remove(pokemon)
                    break
    pokemon_data = []
    for pokemon in pokemon_table:
        pokemon_status = []
        for td_id, td in enumerate(pokemon.find_all("td")):
            if td_id == 2:
                divs = td.find_all("div")
                if len(divs) == 1:
                    pokemon_status += [divs[0].text, "None"]
                else:
                    pokemon_status += [divs[0].text, divs[1].text]
            elif td_id == 3:
                continue
            else:
                pokemon_status += [td.text]
        pokemon_data.append(pokemon_status)
    pokemon_data = pd.DataFrame(
        pokemon_data,
        columns=["#", "Name", "Type1", "Type2", "HP", "Atk", "Def", "SepAtk", "SepDef", "Speed", "Sum"])
    print("finish data formatting, start saving data as file name:{}".format(file_name))
    pokemon_data.to_csv(file_name, index=False, encoding=encoding)
    return pokemon_data


def convert_p2rgb():
    """
    画像ファイルのモードがPなのでRGBに変換する。
    """
    print("change mode ...")
    for i in tqdm(range(1, POKEMON_NUMBER)):
        load_path = "./pokemon_img/{}/yakkun_org.png".format(i)
        save_path = "./pokemon_img/{}/yakkun.png".format(i)
        im = Image.open(load_path)
        if im.mode == "RGB":
            print("画像のモードが既にRGBなので特別な処理はいりません。終了します")
            break
        new_im = im.convert("RGB")
        new_im.save(save_path, quality=100)
        load_path = "./pokemon_img/{}/daisuki_org.png".format(i)
        save_path = "./pokemon_img/{}/daisuki.png".format(i)
        im = Image.open(load_path)
        new_im = im.convert("RGB")
        new_im.save(save_path, quality=100)


def reverse():
    """
    画像を左右・上下反転させる
    """
    print("画像の左右反転を行っています")
    for i in tqdm(range(1, POKEMON_NUMBER)):
        load_path = "./pokemon_img/{}/daisuki.png".format(i)
        save_path = "./pokemon_img/{}/yreverse_d.png".format(i)
        im = cv2.imread(load_path)
        new_im = cv2.flip(im, 1)
        cv2.imwrite(save_path, new_im)
        save_path = "./pokemon_img/{}/xreverse_d.png".format(i)
        new_im = cv2.flip(im, 0)
        cv2.imwrite(save_path, new_im)
        load_path = "./pokemon_img/{}/yakkun.png".format(i)
        save_path = "./pokemon_img/{}/yreverse_y.png".format(i)

        im = cv2.imread(load_path)
        new_im = cv2.flip(im, 1)
        cv2.imwrite(save_path, new_im)
        save_path = "./pokemon_img/{}/xreverse_y.png".format(i)
        new_im = cv2.flip(im, 0)
        cv2.imwrite(save_path, new_im)


def gray_scale():
    """
    グレースケール画像を生成
    """
    print("グレースケール画像を生成しています...")
    for i in tqdm(range(1, POKEMON_NUMBER)):
        load_path = "./pokemon_img/{}/daisuki.png".format(i)
        save_path = "./pokemon_img/{}/grayscale_d.png".format(i)
        im = cv2.imread(load_path)
        new_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(save_path, new_im)
        load_path = "./pokemon_img/{}/yakkun.png".format(i)
        save_path = "./pokemon_img/{}/grayscale_y.png".format(i)
        im = cv2.imread(load_path)
        new_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(save_path, new_im)


def change_contrast(min_ct=50, max_ct=205):
    """
    トーンカーブを用いた調整
    """
    print("コントラストの調整を行っています...")
    diff_ct = max_ct - min_ct

    LUT_HC = np.arange(256, dtype="uint8")
    LUT_LC = np.arange(256, dtype="uint8")

    mask_min = LUT_HC < min_ct
    mask_max = LUT_HC > max_ct
    mask_medium = (LUT_HC >= min_ct) & (LUT_HC <= max_ct)
    LUT_HC[mask_min] = 0
    LUT_HC[mask_max] = 255
    LUT_HC[mask_medium] = 255 * (LUT_HC[mask_medium] - min_ct) / diff_ct

    LUT_LC = min_ct + LUT_LC * (diff_ct) / 255

    for i in tqdm(range(1, POKEMON_NUMBER)):
        load_path = "./pokemon_img/{}/daisuki.png".format(i)
        save_path = "./pokemon_img/{}/hc_d.png".format(i)
        im = cv2.imread(load_path)
        new_im = cv2.LUT(im, LUT_HC)
        cv2.imwrite(save_path, new_im)
        load_path = "./pokemon_img/{}/yakkun.png".format(i)
        save_path = "./pokemon_img/{}/hc_y.png".format(i)
        im = cv2.imread(load_path)
        new_im = cv2.LUT(im, LUT_HC)
        cv2.imwrite(save_path, new_im)


def gamma_filter(high_gamma=1.5, low_gamma=0.75):
    """
    ガンマ補正を用いてコントラストの調整
    """
    LUT_HG = np.arange(256, dtype="uint8")
    LUT_LG = np.arange(256, dtype="uint8")

    LUT_HG = 255 * np.power(LUT_HG / 255, 1 / high_gamma)
    LUT_LG = 255 * np.power(LUT_LG / 255, 1 / low_gamma)

    print("ガンマ補正をかけています...")
    print("High Contrast Gamma = {}".format(high_gamma))
    print("Low Contrast Gamma = {}".format(low_gamma))
    for i in tqdm(range(1, POKEMON_NUMBER)):
        load_path = "./pokemon_img/{}/daisuki.png".format(i)
        save_path = "./pokemon_img/{}/hg_d.png".format(i)
        im = cv2.imread(load_path)
        new_im = cv2.LUT(im, LUT_HG)
        cv2.imwrite(save_path, new_im)
        save_path = "./pokemon_img/{}/lg_d.png".format(i)
        new_im = cv2.LUT(im, LUT_LG)
        cv2.imwrite(save_path, new_im)
        load_path = "./pokemon_img/{}/yakkun.png".format(i)
        save_path = "./pokemon_img/{}/hg_y.png".format(i)
        im = cv2.imread(load_path)
        new_im = cv2.LUT(im, LUT_HG)
        cv2.imwrite(save_path, new_im)
        save_path = "./pokemon_img/{}/lg_y.png".format(i)
        new_im = cv2.LUT(im, LUT_LG)
        cv2.imwrite(save_path, new_im)


def gaussian_filter(filter_size=(5, 5), scale_1=50, scale_2=0):
    """
    ガウシアンフィルターで平均化する
    """
    print("ガウシアンフィルターによる平滑化を行います...")
    print("sigma 1 = {}, sigma 2 = {}".format(scale_1, scale_2))
    for i in tqdm(range(1, POKEMON_NUMBER)):
        load_path = "./pokemon_img/{}/daisuki.png".format(i)
        save_path = "./pokemon_img/{}/gauss_d.png".format(i)
        im = cv2.imread(load_path)
        new_im = cv2.GaussianBlur(im, filter_size, scale_1)
        cv2.imwrite(save_path, new_im)
        load_path = "./pokemon_img/{}/yakkun.png".format(i)
        save_path = "./pokemon_img/{}/gauss_y.png".format(i)
        im = cv2.imread(load_path)
        new_im = cv2.GaussianBlur(im, filter_size, scale_2)
        cv2.imwrite(save_path, new_im)


def gaussian_noise(mean=0, sigma=10):
    """
    ガウシアン分布に基づくノイズを加える
    """
    print("ガウシアン分布に基づくノイズを加えています...")
    print("sigma = {}".format(sigma))
    for i in tqdm(range(1, POKEMON_NUMBER)):
        load_path = "./pokemon_img/{}/daisuki.png".format(i)
        save_path = "./pokemon_img/{}/gaussnoise_d.png".format(i)
        im = cv2.imread(load_path)
        gauss = np.random.normal(mean, sigma, im.size).reshape(im.shape)
        new_im = im + gauss
        cv2.imwrite(save_path, new_im)
        load_path = "./pokemon_img/{}/yakkun.png".format(i)
        save_path = "./pokemon_img/{}/gaussnoise_y.png".format(i)
        im = cv2.imread(load_path)
        gauss = np.random.normal(mean, sigma, im.size).reshape(im.shape)
        new_im = im + gauss
        cv2.imwrite(save_path, new_im)


def type_dummy(pokemon_data):
    type1_dt = pd.get_dummies(pokemon_data["Type1"])
    type2_dt = pd.get_dummies(pokemon_data["Type2"])
    type2_dt[type1_dt.columns] += type1_dt
    pokemon_data = pd.concat([pokemon_data, type2_dt], axis=1)
    pokemon_data.to_csv("pokemon_data.csv", index=False)
    return pokemon_data


def resize_image(resize_shape=(96, 96), load_path=None, save_path=None):
    """
    画像の大きさが異なるので小さい方の大きさ(96,96)に合わせる
    """
    print("画像をリサイズしています...")
    print("new size = ({},{})".format(resize_shape[0], resize_shape[1]))
    if load_path is None and save_path is None:
        for i in tqdm(range(1, POKEMON_NUMBER)):
            load_path = "./pokemon_img/{}/daisuki_org.png".format(i)
            save_path = "./pokemon_img/{}/daisuki.png".format(i)
            im = Image.open(load_path)
            im = im.resize(resize_shape)
            im.save(save_path)
    else:
        im = Image.open(load_path)
        im = im.resize(resize_shape)
        im.save(save_path)


parser = argparse.ArgumentParser(description="This file is used to prepare pokemon image and data augumentation")
parser.add_argument("-di", "--download_image", help="flag to download images", action="store_true")
parser.add_argument("-dd", "--download_data", help="flag to download type data", action="store_true")
parser.add_argument("-r", "--reverse", help="flag to make reverse images", action="store_true")
parser.add_argument("-g", "--gray_scale", help="flag to make gray scale images", action="store_true")
parser.add_argument("-c", "--change_contrast", help="flag to change contrast of the images", action="store_true")
parser.add_argument("-gamma", help="flag to use gamma filter on images", action="store_true")
parser.add_argument("-gf", "--gaussian_filter", help="flag to use gaussian filter on images", action="store_true")
parser.add_argument("-gn", "--gaussian_noise", help="flag to add gaussian noise on images", action="store_true")
parser.add_argument("-resize", help="flag to resize images", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.download_image:
        print("start downloading pokemon image ...")
        get_pokemon_image(url_type="daisukiclub")
        get_pokemon_image(url_type="yakkun")
    if args.download_data:
        type_dummy(get_pokemon_data())
    convert_p2rgb()
    if args.resize:
        resize_image()
    if args.reverse:
        reverse()
    if args.gray_scale:
        gray_scale()
    if args.change_contrast:
        change_contrast()
    if args.gamma:
        gamma_filter()
    if args.gaussian_filter:
        gaussian_filter()
    if args.gaussian_noise:
        gaussian_noise()
