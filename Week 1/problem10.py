def sortArray(Array):
    Negative_arr= []
    Positive_arr = []
    sorted_array = []
    for i in Array:
        if(i<0):
            Negative_arr.append(i)
        else:
            Positive_arr.append(i) 
    minimum_length = min(len(Negative_arr),len(Positive_arr))
    Negative_arr = sorted(Negative_arr)
    Positive_arr = sorted(Positive_arr)
    print(Negative_arr)
    print(Positive_arr)
    for i in range(minimum_length):
        sorted_array.append(Negative_arr[i])
        sorted_array.append(Positive_arr[i])
    if (len(Negative_arr) > minimum_length):
        sorted_array.extend(Negative_arr[minimum_length:])
    if (len(Positive_arr)> minimum_length):
        sorted_array.extend(Positive_arr[minimum_length:])
    return sorted_array            
Array = [10,-1,9,20,-3,-8,22,9,7]
print(sortArray(Array))