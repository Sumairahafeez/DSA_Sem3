def RadixSort(Array):
    # Get the maximum value of Array and declare the exponent = 1
    maxValue = max(Array) 
    exponent = 1
    #while you get most significant bit continue the sort
    while maxValue//exponent > 0:
        # Given algorithm is almost same as counting sort
        n = len(Array)
        output = [0]*10
        count = [0]*10
        for i in range(n):
            index = (Array[i]//exponent)%10
            count[index] += 1
        for i in range(1,10):
            
            count[i]+=count[i-1]
        for i in range(n):
            index = Array[i]//exponent
            count[index % 10] += 1
        for i in range(n-1,-1,-1):
            index = Array[i]//exponent
            count_val = index % 10
            print(f"Count array: {count_val}")
            print(count[count_val]-1)
            output[count[count_val]-1]=Array[i]
            count[count_val] -=1 
        for i in range(n):
            Array[i] = output[i]  
        exponent *= 10  # move the significant bits now                    
def CountingSort(input):
    for i in range(0,len(input)): # for loop to get all numbers as positive in an array
        input[i] = input[i]+max(input)
    RangeOfElements = max(input)# get the maximum number as the range of count array
    print(RangeOfElements)
    CountArray = [0]*(RangeOfElements+1)#get a count array
    OutputArray = [0]*len(input)#get an output array    
    for i in input:
        CountArray[i] = CountArray[i]+1#count the elements
    for i in range(1,RangeOfElements):
        CountArray[i] += CountArray[i-1]#sum the elements in count array
    for i in reversed(input):
        OutputArray[CountArray[i]]=i#get the output array at the index of element
        CountArray[i] -=1
    for i in range(len(input)):
        input[i] = OutputArray[i]#replace the input array with output array
    for i in range(0,len(input)):
        input[i] = input[i]-10 #get the elements of input array as original elements
def BucketSort(Array):
    numBuckets = len(Array)#get number of buckets according to the length of Array
    ResultArray = [] 
    Buckets =  [[]for _ in range(numBuckets)]#select the number of Buckets
    for i in Array:
        index = int(i*numBuckets)#index is equal to the number multiply by the number of Buckets
        Buckets[index].append(i)
    for bucket in Buckets:
        bucket.sort()# sort the elements of bucket
    for i in Buckets:
        ResultArray.extend(i)#store the sorted result in result array
    return ResultArray    
Array = [-5,-10,0,-3,8,5,-1,10]
CountingSort(Array)
print(len(Array))
print(Array)
Array2 = [110,45,65,50,90,602,24,2,66]
RadixSort(Array2)
print(Array2)
Array3 = [0.897, 0.565, 0.656,0.1234, 0.665, 0.3434]

print(BucketSort(Array3))
