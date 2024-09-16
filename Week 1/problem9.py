def PalindromeRecursive(string):
    if(len(string) <= 1):
        return True
    if(string[0] != string[-1]):
            return False
    return PalindromeRecursive(string[1:-1])  
string = input("Enter a string: ")
if(PalindromeRecursive(string)):
    print("Palindrome")
else:
    print("Not Palindrome")
