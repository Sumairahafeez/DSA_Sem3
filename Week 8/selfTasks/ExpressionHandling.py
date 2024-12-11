import re
def infix_to_postfix(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    stack = []
    output = []
    for char in expression:
        if char.isnumeric():
            output.append(char)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        else:
            while stack and stack[-1] != '(' and precedence[char] <= precedence[stack[-1]]:
                output.append(stack.pop())
            stack.append(char)
    while stack:
        output.append(stack.pop())
    return ''.join(output)

def evaluate_postfix(expression):
    stack = []
    for char in expression:
        if char.isnumeric():
            stack.append(int(char))
        else:
            b = stack.pop()
            a = stack.pop()
            if char == '+':
                stack.append(a + b)
            elif char == '-':
                stack.append(a - b)
            elif char == '*':
                stack.append(a * b)
            elif char == '/':
                stack.append(a / b)
            elif char == '^':
                stack.append(a ** b)
    return stack[0]

def chatbot_response(query):
    # Extract the mathematical expression from the query
    expression = re.findall(r'\d+|\+|\-|\*|\/|\^|\(|\)', query)
    expression = ''.join(expression)
    
    # Convert infix to postfix
    postfix_expression = infix_to_postfix(expression)
    
    # Evaluate the postfix expression
    result = evaluate_postfix(postfix_expression)
    
    return result

# Example usage
query = "What is (1+0)*5^2"
print(chatbot_response(query))  # Output: 25