def Partition(Arr,p,r):
    # Get last element as pivot
    pivot = Arr[r] 
    # Get lower index as i which is the index of first element in the array
    lowerIndex = p-1
    # starts the loop from 0 to 7
    for j in range(p,r):
        # if array element is less than pivot increase the lower index and place the pivot at that index
        if Arr[j]<=pivot:
            lowerIndex = lowerIndex+1
            Arr[lowerIndex],Arr[j] = Arr[j],Arr[lowerIndex]
    Arr[lowerIndex+1],Arr[r] = Arr[r],Arr[lowerIndex+1]
    return lowerIndex+1;        
def QuickSort(Arr,p,r):
    # Base Case if starting is less than end point get pivot and apply quicksort on two points
    if(p<r): 
        q = Partition(Arr,p,r)
        print(q)
        print(Arr)
        QuickSort(Arr,p,q-1)
        QuickSort(Arr,q+1,r)
#Get the Array and Check the sorting 
Array = [30,50,15,25,80,20,90,45]
r = len(Array)-1
(QuickSort(Array,0,r))
print(Array)    