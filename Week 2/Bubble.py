import time 
import funcs
# Enter the FileName to store the sorted Arrays
fileName = 'Bubble.txt' 
size = int(input("Enter the size of the array: "))
# Generate an array of Random Numbers of Given Size
Array = funcs.RandomArray(size)
# Get starting and ending Index
start = int(input("Enter the starting index to sort the array: "))
end = int(input("Enter the ending index to sort the array: "))
print("Actual Array: ",Array)
# Get Time before starting the sorting and After sorting
start_time = time.time()
ans= funcs.BubbleSort(Array,start,end)
end_time = time.time()
# Get the difference of time and print the result
runtime = end_time - start_time
print("Sorted Array: ",ans) 
print("Time Taken",runtime,"seconds")
# Store the Data in the file
funcs.StoreArray(Array,fileName)

