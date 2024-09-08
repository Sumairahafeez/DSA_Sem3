def StringReverse(str,starting,ending):
    str2 = str[starting:ending]
    return str2[::-1] 
s = input("Enter a string: ")
starting = int(input("Enter the starting portion: "))
ending = int(input("Enter the ending portion: "))
print(StringReverse(s,starting,ending))