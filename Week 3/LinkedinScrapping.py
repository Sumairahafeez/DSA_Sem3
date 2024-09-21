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
Location = []
Company = []
JobInsight = []
TimeUploaded = []


driver.get("https://www.linkedin.com/jobs/")

content = driver.page_source

soup = BeautifulSoup(content, features="html.parser")
# print(soup)
for a in soup.findAll("li", attrs={"class": "discovery-templates-entity-item"}):
    print(a)
    JobTitle = a.find("a",attrs = {"id":"ember288"})
    JobLocation = a.find("span",attrs = {"class":"job-card-container__primary-description"})
    CompanyData = a.find("li",attrs = {"class":"job-card-container__metadata-item"})
    Insight = a.find("div",attrs = {"class":"job-card-container__job-insight-text"})
    Time = a.find("li",attrs = {"class":"job-card-container__footer-item"})

    if JobTitle != None and JobLocation != None and CompanyData != None :
        Title.append(JobTitle.text)
        Location.append(JobLocation.text)
        Company.append(CompanyData.text)
        JobInsight.append(Insight.Text)
        TimeUploaded.append(Time.text)
    if len(Title) == 10:
        break

df = pd.DataFrame({"Job Titel": Title, "Job Location": Location, "Company": Company, "Insights":JobInsight, "Time":TimeUploaded })
df.to_csv("Linkedin.csv", encoding="utf-8")