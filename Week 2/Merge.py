import funcs
import time
def main():
    # Enter the File Name 
    fileName = 'Merge.txt'
    # Get the Size and genearte random array of gievn size
    size = int(input("Enter the size of the array: "))
    Array = funcs.RandomArray(size)
    start = int(input("Enter the starting index to sort the array: "))
    end = int(input("Enter the ending index to sort the array: "))
    print("Actual Array: ",Array)
    # Get the start and end time of the Sorting
    start_time = time.time()
    ans = funcs.MergeSort(Array,start,end)
    end_time = time.time()
    runtime = end_time - start_time
    # Print the result and store the sorted array in file
    print("Sorted Array: ",ans) 
    print("Time Taken",runtime,"seconds")
    funcs.StoreArray(Array,fileName)
main()

