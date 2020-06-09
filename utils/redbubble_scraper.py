import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import base64
import re
import urllib.request
import html
import json
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import selenium as se
import os
import os.path
import ast

class Clothing:
    def __init__(self, url, id, dir="all_mens/"):
        self.id = id
        self.painting = None
        self.pictures = []
        self.designer = None
        self.type = None
        self.painting_name = None
        self.painting_artist = None
        self.url = url
        self.dir = dir
        self.set_params()

    def set_params(self):
        """
            Set all the parameters of the Dress object.
        """
        paint = False
        try:
            # start a ghostdriver to load javascript loaded images

            #driver = webdriver.PhantomJS()
            options = Options()
            options.headless = True
            options.add_argument('--dns-prefetch-disable')
            driver = webdriver.Chrome('/usr/lib/chromium-browser/chromedriver', chrome_options=options)
            #driver.set_window_size(1200, 600)
            driver.set_page_load_timeout(20)
            driver.get(self.url)
            #driver.manage().timeouts().implicitlyWait(500, TimeUnit.MILLISECONDS);
            html = driver.page_source
            # Set the correct names
            #with open("html.txt", "a") as f:
            #    f.write(html)
            title = driver.find_element_by_class_name("ArtworkDetails__workTitle--3hle8").text
            self.designer = driver.find_element_by_class_name("ArtworkDetails__artistLink--1L81W").text
            if self.designer == "":
                des_catch = title.split('by ')
                if des_catch[1]:
                    self.designer = des_catch[1]

            self.type = driver.find_element_by_class_name("ProductConfiguration__description--1ZVD_").text
            
            # Extract painting artist and painting name
            title = Clothing.removeNonAscii(title)
            for char in [self.designer, ' by ']:
                title = title.replace(char, '')
            title = re.sub(r"[^a-zA-Z0-9\s]", "", title)
            title = title.replace('  ', ' ').replace(' ', '_')
            if '_-_' in title:
                self.painting_artist = title.split('_-_',1)[0]
                self.painting_name = title.split('_-_',1)[1]
            else:
                self.painting_name = title

            # Extracting the images
            img_id = 0
            for result in driver.find_elements_by_class_name("GalleryImage__img--12Vov"):
                image_src = result.get_attribute("src")
                if image_src:
                    if len(image_src) > 200:
                        self.download_base64(img_id, src=image_src)
                        img_id += 1
                    else:
                        self.download_url_images(image_src, paint, id=img_id)
                        img_id += 1
                        paint = True
            driver.close()
            driver.quit()
        except Exception as e:
            print('Failed!!!!!')
            print(e)
            with open("failed.txt", "a") as f:
                f.write('-------------------------\n')
                f.write(self.url + '\n')
                #f.write(' ' + e + '\n')
                return

    def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)



    def download_base64(self, id, src=None,soup = None):
        """
            Takes in a BeautifulSoup object and optionaly a directory name.

            Searches the BeautifulSoup object for a specific class which is associated
            with base64 images. Encodes the retrieved string to bytes and writes to img
            file in the 'images' directory
        """
        if soup:
            image_64 = soup.find_all(class_="GalleryImage__img--12Vov")[0]['src']
        if src:
            image_64 = src
        image_data = bytes(image_64, encoding='utf-8')
        file_name = self.dir + str(self.id) + '_' + self.painting_name + "_big_"+ str(id) + ".png"

        with open(file_name, "wb") as fh:
            fh.write(base64.decodebytes(image_data[21:]))
            self.pictures.append(file_name)

    def download_url_images(self, image_urls, paint, id):
        """
            Takes in either a list or single url.

            Loops over the given urls and tries to save the picture from the site to
            the given directory or if no directory spicified, to the images directory.
        """
        image_types = ["jpg", "png"]
        ext = image_urls[-3:]
        if ext in image_types:
            if paint:
                file_name = self.dir + str(self.id) + '_' + self.painting_name + "_" + str(id) + "." + ext
            else:
                file_name = self.dir + str(self.id) + '_' + self.painting_name + "_" + str(id) + "." + ext
            urllib.request.urlretrieve(image_urls, file_name)
            if paint:
                self.painting = file_name
                self.pictures.append(file_name)
            else:
                self.pictures.append(file_name)

    def __str__(self):
        return "Dress name: {}, Painting: {}, Image count: {}, Url: {}".format(self.name, self.painting_name, len(self.pictures), self.url)


def main():
    dresses = []
    urls = []
    id = 0
    # all_mens last page = 441
    # all_mens_url = "https://www.redbubble.com/shop/mens-clothes?cat_context=men-clothes&page=" + str(page_index) + "&rbs=f973a9c6-8840-466c-9efb-2a0c9010ddc8"
    all_mens_last = 441
    class_name = 'styles__link--2pzz4' #"shared-components-ShopSearchResultsGrid-ShopSearchResultsGrid__searchResultItemContainer--qmOnC"
    # womens_dresses_url last page = 32
    # womens_dresses_url = 'https://www.redbubble.com/shop/famous+painting+dresses?cat_context=w-dresses&page=' + str(page_index) + '&rbs=e25abfbe-12dd-43e1-94b5-53cba74e0598'
    womens_dresses_last = 32
    # Append all 32 (or less for testing) dress-showcase urls to list
    for page_index in range(60, all_mens_last):
        #https://www.redbubble.com/shop/mens-baseball-sleeve-tshirts?cat_context=m-tees&page=2&rbs=595c2341-20ee-4644-b6d3-4df7704d7f10
        urls.append('https://www.redbubble.com/shop/mens-baseball-sleeve-tshirts?cat_context=m-tees&page=' + str(page_index) + '&rbs=595c2341-20ee-4644-b6d3-4df7704d7f10')
        #urls.append("https://www.redbubble.com/shop/mens-clothes?cat_context=men-clothes&page=" + str(page_index) + "&rbs=f973a9c6-8840-466c-9efb-2a0c9010ddc8" )
    # Loop over showcase pages, scrape all dresses and write their title attributes
    # to a file.
    for url in urls:
        dresses = []
        print('Url ', url)
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        for dress_url in soup.find_all("a", class_=class_name):
            # if id < 859 and id > 809:
            dress_url = 'https://www.redbubble.com' + dress_url['href']
            print('dress url ', dress_url)
            the_dress = Clothing(dress_url, id)
            # dresses.append(Dress(dress_url, id))
            id += 1

            with open("t-shirt_again.json", 'a') as file:
                # for dress in dresses:
                file.write(json.dumps(the_dress.__dict__) + '\n')
            # elif id >=859:
            #     return
            # else:
            #     id += 1


def retry_failed():
    with open('failed.txt') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in open('filename')]

    #for line

if __name__ == "__main__":
    main()
