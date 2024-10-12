import time
list = []#declare a list
for i in range(1,100):
    # check time
    startTime = time.time()
    list.append(i)
    endTime = time.time()
    totalTime = endTime-startTime
    # append and print the time
    print(f'It took {totalTime} to append {i} in list')