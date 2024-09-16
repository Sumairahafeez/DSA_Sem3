import funcs
import time
def Sort(words):
    # Code to sort the words array using Insertion and Merge Sort
    insertionStart = time.time()
    ans = funcs.InsertionSort(words,0,len(words))
    insertionEnd = time.time()
    MergeStart = time.time()
    words = funcs.ShuffleArray(words,0,len(words))
    MergeEnd = time.time()
    runtime = insertionEnd-insertionStart
    runtime1 = MergeEnd - MergeStart
    print(ans)
    print('Insertion Sort is done in: ',runtime)
    print('Merge Sort is done in: ',runtime1)
    
def Main():
    fileName = 'words.txt'
    words = funcs.ReadNumbers(fileName)
    Sort(words)
    (funcs.ShuffleArray(words,0,len(words)))
    Sort(words)
Main()
