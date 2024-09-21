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
Product1 = []
Product2 = []
Product3= []
Product4 = []


driver.get("https://www.amazon.com/?&tag=googleglobalp-20&ref=pd_sl_7nnedyywlk_e&adgrpid=159651196451&hvpone=&hvptwo=&hvadid=675114638556&hvpos=&hvnetw=g&hvrand=18193258499150172538&hvqmt=e&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1011082&hvtargid=kwd-10573980&hydadcr=2246_13649807")

content = driver.page_source

soup = BeautifulSoup(content, features="html.parser")
# print(soup)
for a in soup.findAll("div", attrs={"class": "a-cardui"}):
    print(a)
    CategoryName = a.find("div",attrs = {"class":"a-cardui-header"})
    Product1Name = a.find("span",attrs = {"class":"a-size-small"})
    Product2Name = a.find("span",attrs = {"class":"a-size-small"})
    Product3Name = a.find("span",attrs = {"class":"a-size-small"})
    Product4Name = a.find("span",attrs = {"class":"a-size-small"})

    if Product1 != None and Product2 != None and Product3 != None and Product4 != None :
        Category.append(CategoryName.text)
        Product1.append(Product1Name.value) 
        Product2.append(Product2Name.text)
        Product3.append(Product3Name.Text) 
        Product4.append(Product4Name.Text)
    if len(Category) == 10:
        break

df = pd.DataFrame({"Product Category": Category, "Product1": Product1, "Product2": Product2, "Product3":Product3, "Product4":Product4 })
df.to_csv("Amazon.csv", encoding="utf-8")