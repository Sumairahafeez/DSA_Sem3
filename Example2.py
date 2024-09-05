#Recursive problems
#Example 2.1 Sum of numbers using recursive methods
def sum(n):
    if n==0:
        return n;
    else:
        return n + sum(n-1);
print(sum(10))
#Example 2.2 Print array of elements
def Print_array(arr,start,end):
    if start==end:
        print(arr[start])
    else:
        print(arr[start])
        Print_array(arr,arr[start+1],end)
arr = [0,1,2,3,4,5,6,7,8,9,10]
print(Print_array(arr,0,len(arr)-1))
#Example 2.3 Power rule
def power (n, k):
    if k == 1:
        return n
    else:
        return n * power (n, k-1)

print(power(3))