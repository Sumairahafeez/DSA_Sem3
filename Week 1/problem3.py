def Minimum(Array,starting,ending):
    Array2 = Array[starting:ending+1]
    print(Array2)
    minim = min(Array2)
    print(minim)
    for i in range(len(Array)):
        if(minim == Array[i]):
            return i 
Arrayrange = int(input("Enter the range of your array: "));
Array = []
for i in range(0,Arrayrange):
    Array.insert(i,input("Enter the element: "));
starting_index = int(input("Enter the starting index: "));
ending_index = int(input("Enter the ending index: "));
print(Minimum(Array,starting_index,ending_index))