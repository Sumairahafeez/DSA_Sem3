from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.chrome.service import Service
import time as sleep

# webdriver can be downloaded from
# https://sites.google.com/chromium.org/driver/downloads/versionselection?authuser=0

service = Service(executable_path="F:\\3rd Sem\\DSA Lab\\Week 3\\chromedriver-win64\\chromedriver.exe")
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)
# driver = webdriver.Chrome(executable_path='C:/Program Files/chromedriver-win64/chromedriver.exe')
title = []
products = []  # List to store name of the product
prices = []  # List to store price of the product
# ratings = []  # List to store rating of the product

driver.get("https://thelastwordbks.com/")

content = driver.page_source

soup = BeautifulSoup(content, features="html.parser")
# print(soup)
for a in soup.findAll("div", attrs={"class": "innerer"}):
    print(a)
    name = a.find("div", attrs={"class":"title"})
    price = a.find("span", attrs={"class": "theme-money"})
    if title != None and price != None:
        title.append(name.text)
        prices.append(price.text)
    if len(products) == 56:
        break

df = pd.DataFrame({"Book Name": title, "Price": prices})
df.to_csv("Book.csv", encoding="utf-8")