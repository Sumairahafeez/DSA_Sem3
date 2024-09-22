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
Professor = []
Description = []
Session= []
Books= []


driver.get("https://github.com/search?q=webdev&type=repositories")

content = driver.page_source

soup = BeautifulSoup(content, features="html.parser")
# print(soup)
for a in soup.findAll("div", attrs={"class": "gZKkEq"}):
    print(a)
    CourseName = a.find("a",attrs = {"class":"dIlPa"})
    # Teacher = a.find("h7")
    # Desc = a.find("h7")
    # session = a.find("p",attrs = {"class":"card-text"})
    # href = a.find("a",attrs = {"class":"btn"})
    # if CourseName != None and Teacher != None and Desc != None and session != None :
    Name.append(CourseName.text)
        # Professor.append(Teacher.value) 
        # Description.append(Desc.text)
        # Session.append(session.Text) 
        # Books.append(href.href)
    if len(Name) == 10:
        break

df = pd.DataFrame({"Name": Name})
df.to_csv("Eduko.csv", encoding="utf-8")