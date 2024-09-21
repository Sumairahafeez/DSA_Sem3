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
Title = []
Stars = []
RegularPrice = []
SalePrice = []
status = []
imageURL = []


driver.get("https://missroseofficial.com.pk/collections/blushes-bronzers")

content = driver.page_source

soup = BeautifulSoup(content, features="html.parser")
# print(soup)
for a in soup.findAll("div", attrs={"class": "card"}):
    print(a)
    ProductTitle = a.find("a",attrs = {"class":"full-unstyled-link"})
    ProductStars = a.find("span",attrs = {"class":"loox-rating-label"})
    ProductPrice = a.find("span",attrs = {"class":"price-item"})
    ProductStatus = a.find("span",attrs = {"class":"card__badge"})
    Image = a.find("div",attrs = {"class":"media"})

    # if ProductTitle != None and ProductPrice != None and Image != None and ProductStars != None :
    Title.append(ProductTitle.text)
    Stars.append(ProductStars.text) if ProductStars != None else Stars.append(None)
    SalePrice.append(ProductPrice.text)
    status.append(ProductStatus.Text) if ProductStatus != None else status.append(None)
    imageURL.append(Image.srcset)
    if len(Title) == 10:
        break

df = pd.DataFrame({"Product Title": Title, "Stars": Stars, "SalePrice": SalePrice, "status":status, "imageURL":imageURL })
df.to_csv("Linkedin.csv", encoding="utf-8")