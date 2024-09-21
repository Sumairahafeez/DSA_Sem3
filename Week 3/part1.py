# Bar Chart
import matplotlib.pyplot as plt
import pandas as pd 
df = pd.read_csv('population_by_country_2020.csv')
print(df.dtypes)
list1_country = df['Country or (dependency)'].values.tolist()
list2_population = df['Population (2020)'].values.tolist()
plt.bar(list1_country,list2_population,width=2,color = ['red','green'])
plt.show()
# Line Chart
data = pd.read_csv('dailyActivity_merged.csv')
List1_steps = data['TotalSteps'].values.tolist()
List2_date = data['ActivityDate'].values.tolist()
plt.plot(List2_date,List1_steps)
plt.show()
# Total Daily Distance Covered
data = pd.read_csv('dailyActivity_merged.csv')
List1_Distance = data['TotalDistance'].values.tolist()
List2 = data['ActivityDate'].values.tolist()
plt.plot(List2,List1_Distance)
plt.show()
# Total Time in Bed
data = pd.read_csv('dailyActivity_merged.csv')
List1 = data['SedentaryMinutes'].values.tolist()
List2 = data['ActivityDate'].values.tolist()
plt.scatter(List2,List1)
plt.show()
# Hourly steps on 12th April 2016
data = pd.read_csv('dailyActivity_merged.csv')
List1 = data['TotalSteps'].values.tolist()
Total_steps = List1[1]
plt.pie(Total_steps)
plt.show()