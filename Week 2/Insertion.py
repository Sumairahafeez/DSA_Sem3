import time
import funcs     
def Main():
    # Enter the FileName
    fileName = 'Insertion.txt'
    # Get Input and generate a random array of given size
    size = int(input("Enter the size of the array: "))
    Array = funcs.RandomArray(size)
    start = int(input("Enter the starting index to sort the array: "))
    end = int(input("Enter the ending index to sort the array: "))
    print("Actual Array: ",Array)
    # Get the time of sorting 
    start_time = time.time()
    ans= funcs.InsertionSort(Array,start,end)
    end_time = time.time()
    runtime = end_time - start_time
    # Print the given result and store it in file
    print("Sorted Array: ",ans) 
    print("Time Taken",runtime,"seconds")
    funcs.StoreArray(Array,fileName)
Main()        