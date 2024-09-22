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
Category = []
PriceRange = []
minOrder = []
Location= []
Time = []


driver.get("https://www.alibaba.com/trade/search?spm=a27aq.cp_3.1835861830.2.702b5b91IDHZwI&categoryId=100005791&SearchText=Casual+Dresses&indexArea=product_en&fsb=y&productId=1600572813689")

content = driver.page_source

soup = BeautifulSoup(content, features="html.parser")
# print(soup)
for a in soup.findAll("div", attrs={"class": "card-info"}):
    print(a)
    Name = a.find("h2",attrs = {"class":"search-card-e-title"})
    Price = a.find("div",attrs = {"class":"search-card-e-price-main"})
    minOrderValue = a.find("div",attrs = {"class":"search-card-m-sale-features__item"})
    Loc = a.find("a",attrs = {"class":"search-card-e-company"})
    experience = a.find("a",attrs = {"class":"search-card-e-supplier__year"})

    if Name != None and Price != None and minOrder != None and Loc != None :
        Category.append(Name.text)
        PriceRange.append(Price.value) 
        minOrder.append(minOrderValue.text)
        Location.append(Loc.Text) 
        Time.append(experience.Text)
    if len(Category) == 10:
        break

df = pd.DataFrame({"Product Name": Category, "Price Range": PriceRange, "Min Order": minOrder, "Location":Location, "Time Selling":Time })
df.to_csv("AliBaba.csv", encoding="utf-8")