def SortedMerge(Arr1,Arr2):
    sortArray = Arr1+Arr2
    for i in range(len(sortArray)):
        for j in range(0,len(sortArray)-i-1):
            if(sortArray[j]>= sortArray[j+1]):
                sortArray[j],sortArray[j+1] = sortArray[j+1],sortArray[j]
    return sortArray
Array1 = [0,3,4,7,10,11]
Array2 = [1,8,13,24]
print(SortedMerge(Array1,Array2))