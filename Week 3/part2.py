
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
#Main That Gets the Data 
df = pd.read_csv('Train.csv')
Symptomps = df.columns[:-1]
for i in Symptomps:
    PlotData(i)

