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
Name = []
Level = []
Highlight = []
Rating= []
startingPrice= []


driver.get("https://www.fiverr.com/categories/video-animation/animated-characters-modeling?source=vertical-buckets")

content = driver.page_source

soup = BeautifulSoup(content, features="html.parser")
# print(soup)
for a in soup.findAll("div", attrs={"class": "basic-gig-card"}):
    print(a)
    FreelancerName = a.find("span",attrs = {"class":"_1sz16f5k"})
    itsLevel = a.find("p",attrs = {"class":"z58z872"})
    highlight = a.find("p",attrs = {"class":"QTdEgIS"})
    rating = a.find("strong",attrs = {"class":"rating-score"})
    price = a.find("a",attrs = {"class":"hxtGeVp"})

    if FreelancerName != None and price != None and itsLevel != None and rating != None :
        Name.append(FreelancerName.text)
        Level.append(itsLevel.value) 
        Highlight.append(highlight.text)
        Rating.append(rating.Text) 
        startingPrice.append(price.Text)
    if len(Name) == 10:
        break

df = pd.DataFrame({"Name": Name, "Level": Level, "Highlight": Highlight, "Rating":Rating, "starting Price":price })
df.to_csv("FiverVideoAnimation.csv", encoding="utf-8")