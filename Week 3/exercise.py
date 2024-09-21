import matplotlib.pyplot as plt
import pandas as pd
# Load Data From Pandas
# data = pd.read_csv('Train.csv')
# List1 = data.columns[:-1]
# List2 = data['TYPE'].values
# # List1 = data['MUSCLE_ACHES'].values.tolist()
# # List2 = data['TYPE'].values.tolist()
# # Scattered Graph between symptom and the label
# plt.figure(figsize=(15, 10))
# for i,symptomp in enumerate(List1):
#     plt.subplot(5,5,i+1)
#     plt.scatter(List2,symptomp)
# plt.tight_layout()    
# plt.show()

def Plotting(i,symptomp_name,label_name):
    # plt.subplot(5,5,15)
    plt.scatter(symptomp_name[i],label_name[i])
    plt.show() 

data = pd.read_csv('Train.csv')
symptomps_name = data.columns[:-1]
label_name = data.columns[-1]
for i in range(len(symptomps_name)):
    Plotting(i,symptomps_name,label_name)
  
