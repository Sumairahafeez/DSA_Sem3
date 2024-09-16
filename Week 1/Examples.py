import random
#Example 1.1 Displaying a varibale
a = 4;
print("the value of a  is ",a);
#Example 1.2 Input value from a user
a = input("Enter the value ") #It is a string
#conversion of string into int
b = int(a)
print(b)#it will give error so write 
print(str(b))
print(a)
#Example 1.3 Array Declarartion 
array = [];#1D Arrays
#used a for loop to insert numbers in array and print it
for i in range(4):
    array.insert(i,i);
print(array);
#2D arrays
array2 = [[1,2,3],[4,5,6],[7,8,9]] #assigning values simply
print(array2)
#using user assignment
for x in range(3):
    array.insert(x,3);
    array2.insert(x,array);
print(array2)
#Example 1.4 arrays of zeros
array = [0] * 10 #array of length 10 having zeros
array1 = [[0 for x in range(3)]for y in range(4)]
print(array)
print(array1)
#Example 1.5 1D array of random int
arr = []#declare array
min = 0# set min and max for random int
max = 100
n = 5 #start a loop
for i in range(n):
    num = random.randint(min,max);
    arr.append(num)
#print the result
print(arr)
#Example 1.6 Traversing an array
    #traverse in forward direction using for loop
str = ["U","E","T"]
for i in range(len(str)):
    print(str[i])
    #reversing using slicing
array = [1,2,3,4,5,6]
print(array[::-1])
print(str[::-1])
    #traversing using reverse method
str.reverse()
print(str)
    #reverse using for loop
for i in range(len(array)-1,-1,-1):
    print(array[i])
#Example 1.7 slicing method
print(array[:2])#starting sub array
print(array[1:3])#mid sub array
print(array[2:])#ending sub array
#Example 1.8 playing with files
given_file = open('Text.txt',mode='r')
lines = given_file.read()
print(lines.split())
file = open('Text.txt',mode='w')
file.write("Hello")
file2 = open('Text.txt', mode='a')
file2.write("World")
#Example 1.10 Play with functions
def DisplayArray(arr):
    for i in arr:
        print(i)

    
arr = [1,2,3,4,5,6,7]
DisplayArray(arr)
