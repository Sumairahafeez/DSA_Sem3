def SearchA(arr, x):
    x = int(x)#convert it into integar
    array = []#declare an array
    for  i in range(len(arr)):
        if(arr[i] == x):
          array.append(i)
    return array
#Declare the array
Array = [22,2,1,7,11,13,5,2,9]
X = input("Enter the element to be searched: ")
#call the function
print(SearchA(Array,X))