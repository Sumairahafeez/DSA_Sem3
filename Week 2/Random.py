import funcs
import time
size = int(input("Enter the size of the array: "))
start_time = time.time()
ans= funcs.RandomArray(size)
end_time = time.time()
runtime = end_time - start_time
print(runtime,"seconds")
print(ans)     
    