
import pandas as pd
import matplotlib.pyplot as plt
# Function to plot each symptomp with respect to the Disease Type
def PlotData(symptomp):
    Type_name = df['TYPE'].values.tolist()
    Disease_Name = df[symptomp].values.tolist()
    plt.scatter(Type_name,Disease_Name)
    plt.title(symptomp)
    plt.ylabel(symptomp)
    plt.show()
#Function That compares data with Test Data
def TestAndTrain(symptomp):
    Disease_Name = test[symptomp].values.tolist() 
    for i in Disease_Name:
        for j in TestSymptomp:
            if i == j:
                print(f'test row matches with the training row')
            else:
                print("Result doesnot matches")    

#Main That Gets the Data 
df = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
Symptomps = df.columns[:-1]
TestSymptomp = test.columns[:-1]
for i in Symptomps:
    PlotData(i)
TestAndTrain(TestSymptomp)
