import time
import funcs          
def main():
    # Enter the fileName and get its size
    fileName = 'Selection.txt'
    size = int(input("Enter the size of the array: "))
    # Get the Random Array of gievn size and get its starting and ending index
    Array = funcs.RandomArray(size)
    start = int(input("Enter the starting index to sort the array: "))
    end = int(input("Enter the ending index to sort the array: "))
    print("Actual Array: ",Array)
    # Get the starting end ending time of sorting
    start_time = time.time()
    ans= funcs.SelectionSort(Array,start,end)
    end_time = time.time()
    # Print the result and store it in the file
    runtime = end_time - start_time
    print("Sorted Array: ",ans) 
    print("Time Taken",runtime,"seconds")
    funcs.StoreArray(Array,fileName)
main()   