def Minimum(Array,starting,ending):
    Array2 = Array[starting:ending+1]
    minim = min(Array2)
    for i in range(len(Array)):
        if(minim == Array[i]):
            return i 
def Sort4(Array):
    Array3 = []
    for i in range(0,len(Array)):
        index = Minimum(Array,i,len(Array))
        Array3.insert(i,Array[index])
    return Array3
    
Arrayrange = int(input("Enter the range of your array: "));
Array = []
for i in range(0,Arrayrange):
    Array.insert(i,input("Enter the element: "));
starting_index = int(input("Enter the starting index: "));
ending_index = int(input("Enter the ending index: "));
print(Sort4(Array))