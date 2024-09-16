import random
#Function to Get Random Array of given Size
def RandomArray(size):
    Arr = []
    for i in range(0,size):
        Random_Number = random.randint(-15000,15000) 
        Arr.append(Random_Number)
    return Arr
#Storing Array in a single file
def StoreArray(Array,fileName):
    file = open(file=fileName,mode='a',newline='')
    # Read the array and write it in new line using comma seperated
    file.write('[')
    for i in Array:
        file.write(str(i))
        if(i != Array[len(Array)-1]):
            file.write(',')
    file.write(']')  
#Bubble Sort of Array  
def BubbleSort(array,start,end):
    for i in range(start,end+1):#starts the array from start to end
        for j in range(start,end-i-1):#compare that first element with the remaining element
            if(array[j+1]<array[j]):#if the next element is less than first one swap them 
                array[j],array[j+1] = array[j+1],array[j]
    return array
#Function to get Sorting By Insertion Sort
def InsertionSort(Array,start,end):
    for i in range(start,end): # traverrse the array
        Key = Array[i] #set the first element of array as Key
        j = i-1 # check the previous elements 
        while j>=0 and Array[j]>Key:
            Array[j+1] = Array[j]# while previous index is greater than 0 and previous element is greater than key
            j = j-1
        Array[j+1] = Key #set the next index as key and again go through the loop 
    return Array
#Function to get Sort by Selection Sort
def SelectionSort(array,start,end):
    for i in range(start,end):
        min = i # set the minimun index as i 
        for j in range(i+1,end):
            if(array[j]<array[min]):# if the next element is greater then min set min to j
                min = j
        array[i],array[min] =  array[min],array[i] #replace the min element with the given index element
    return array 
#Function to get the sorted Array Merged 
def Merge(Array,p,q,r):
    LeftArray = q-p+1 #left array gets elements less than mid
    rightArray = r-q #it get elements after mid
    Left = Array[p:p + LeftArray]
    Right = Array[q + 1:q + 1 + rightArray]
 
    leftIndex=0
    rightIndex = 0
    ArrayLocation = p #now check both arrays if left element is less than right place it at array location  viceversa
    while leftIndex < LeftArray and rightIndex < rightArray:
        if(Left[leftIndex] <= Right[rightIndex]):
            Array[ArrayLocation] = Left[leftIndex]
            leftIndex+=1
        else:
            Array[ArrayLocation] = Right[rightIndex]
            rightIndex+=1
        ArrayLocation +=1  
        # fill the array with remaining left index  
    while leftIndex < len(Left):
        Array[ArrayLocation] = (Left[leftIndex])
        leftIndex+=1
        ArrayLocation+=1
        # fill the array with right elements that are remaining
    while rightIndex < len(Right):
        Array[ArrayLocation] = (Right[rightIndex])
        rightIndex+=1
        ArrayLocation+=1
              
# #Function to get sorting by MERGE Algorithm
def MergeSort(array,start,end):
    #if start and end has differenece divide the array and merge them
    if(start < end):
        mid = (start+end)//2
        (MergeSort(array,start,mid))
        (MergeSort(array,mid+1,end))
        Merge(array,start,mid,end-1)



def HybridMerge(array,start,end):
   #if the difernce is less than a given threshold 
   if end-start <= 8:
      return( InsertionSort(array,start,end))
   #else follow the merge sort
   else:
        if start<end-1:
            mid = (start+end)//2
            (MergeSort(array,start,mid))
            (MergeSort(array,mid,end))
            Merge(array,start,mid,end)
#function to read size from values.txt file            
def ReadNumbers(fileName):
    size = []
    file = open(fileName,mode='r')
    lines = file.readlines()
    for line in lines:
        size.append(line)
    return size   
#Function to shuffle a gievn array
def ShuffleArray(Array,start,end):
   newArray = Array[start:end]
   random.shuffle(newArray)
   return newArray          