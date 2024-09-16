import funcs
import time
import csv
def main():
    fileName = 'nValues.txt'
    size = funcs.ReadNumbers(fileName)
    #Get the given array with size
    with open('runTime.csv',mode='w',newline='') as csvFile:
        fieldnames = ['Size', 'Bubble Sort', 'Insertion Sort', 'Selection Sort', 'Merge Sort', 'Hybrid Merge Sort']
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
            
            # Write the header
        writer.writeheader()

        for i in range(0,len(size)):
            # Repeat the process for every size 
            Array = funcs.RandomArray(int(size[i]))
            start = 0
            end = int(size[i])
            # GET Start and end time of bubble sort
            start_time = time.time()
            funcs.BubbleSort(Array,start,end)
            end_time = time.time()
            # Insertion Sort
            insertionStart = time.time()
            funcs.InsertionSort(Array,start,end)
            insertionEnd = time.time()
            # Selection Sort
            selectionStart = time.time()
            funcs.SelectionSort(Array,start,end)
            selectionEnd = time.time()
            # Merge Sort
            mergeStart = time.time()
            funcs.MergeSort(Array,start,end)
            mergeEnd = time.time()
            # Hybrid Merge Sort
            HybridStart = time.time()
            funcs.HybridMerge(Array,start,end)
            HybridEnd = time.time()
            # Run Times of All elements
            runtime = end_time - start_time
            runtime1 = insertionEnd-insertionStart
            runtime2 = selectionEnd-selectionStart
            runtime3 = mergeEnd - mergeStart
            runtime4 = HybridEnd - HybridStart
            # print("Sorted Array: ",ans) 
            print("Time Taken for BubbleSort is ",runtime,"seconds for ",int(size[1])," size")
            print("Time Taken for InsertionSort is ",runtime1,"seconds for ",int(size[i])," size")
            print("Time Taken for SelectionSort is ",runtime2,"seconds for ",int(size[i])," size")
            print("Time Taken for MergeSort is ",runtime3,"seconds for ",int(size[i])," size")
            print("Time Taken for Hybrid MergeSort is ",runtime4,"seconds for ",int(size[i])," size")
            writer.writerow({
                    'Size': int(size[i]),
                    'Bubble Sort': runtime,
                    'Insertion Sort': runtime1,
                    'Selection Sort': runtime2,
                    'Merge Sort': runtime3,
                    'Hybrid Merge Sort': runtime4
                })
main()

