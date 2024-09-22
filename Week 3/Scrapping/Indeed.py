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
Company = []
location = []
salary= []
Acitivity= []


driver.get("https://www.fiverr.com/categories/online-marketing/social-marketing?source=hplo_subcat_first_step&pos=5")

content = driver.page_source

soup = BeautifulSoup(content, features="html.parser")
# print(soup)
for a in soup.findAll("div", attrs={"class": "job-seen-beacon"}):
    print(a)
    JobName = a.find("a",attrs = {"class":"jcs-JobTitle"})
    CompanyName = a.find("div",attrs = {"class":"css-1qv0295"})
    loc = a.find("div",attrs = {"class":"css-63koeb"})
    SalarayRange = a.find("div",attrs = {"class":"css-1cvv01b"})
    activityStatus = a.find("span",attrs = {"class":"css-qvloho"})

    if JobName != None and CompanyName != None and loc != None and SalarayRange != None :
        Name.append(JobName.text)
        Company.append(CompanyName.value) 
        location.append(loc.text)
        salary.append(SalarayRange.Text) 
        Acitivity.append(activityStatus.Text)
    if len(Name) == 10:
        break

df = pd.DataFrame({"Name": Name, "Company": Company, "Location": location, "salary":salary, "Activity":Acitivity })
df.to_csv("Indeed.csv", encoding="utf-8")