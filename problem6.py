def SumIterative(number):
    sum = int(0);
    for i in number:
        sum += int(i)
    return sum
def SumRecursion(number):
    number = int(number)
    last_digit = number%10
    remaining_number = int(number//10)
    if number == 0:
        return number
    else:
       return last_digit+SumRecursion(remaining_number)
number = (input("Enter the number: "))
print("Its iterative sum is: ",SumIterative(number))
print("Its recursive sum is: ",SumRecursion(number))
