class ExpressionConverter:
    # Operator precedence
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

    def __init__(self, expression):
        self.expression = expression

    def is_operator(self, c):
        return c in ['+', '-', '*', '/', '^']

    def is_operand(self, c):
        return c.isalnum()

    def infix_to_postfix(self):
        stack = []  # Stack for operators
        postfix = []  # Result

        for char in self.expression:
            if self.is_operand(char):  
                postfix.append(char)
            elif char == '(' or char == '[' or char == '{':
                stack.append(char)
            elif char == ')' or char == ']' or char == '}':
                # Pop until matching opening bracket
                while stack and stack[-1] not in '([{':
                    postfix.append(stack.pop())
                stack.pop()  # Remove the opening bracket
            else:
                # Pop operators with higher precedence
                while stack and stack[-1] not in '([{' and self.precedence[char] <= self.precedence.get(stack[-1], 0):
                    postfix.append(stack.pop())
                stack.append(char)

        while stack:
            postfix.append(stack.pop())  # Pop remaining operators

        return ''.join(postfix)

    def infix_to_prefix(self):
        # Reverse the infix expression and adjust parentheses
        reversed_expression = self.expression[::-1]
        adjusted_expression = reversed_expression.replace('(', 'temp').replace(')', '(').replace('temp', ')')
        adjusted_expression = adjusted_expression.replace('[', 'temp').replace(']', '[').replace('temp', ']')
        adjusted_expression = adjusted_expression.replace('{', 'temp').replace('}', '{').replace('temp', '}')

        # Convert reversed infix to postfix
        postfix = ExpressionConverter(adjusted_expression).infix_to_postfix()

        # Reverse postfix to get prefix
        return postfix[::-1]

    def postfix_to_infix(self, postfix):
        stack = []

        for char in postfix:
            if self.is_operand(char):
                stack.append(char)
            else:
                # Pop two operands and create expression
                op1 = stack.pop()
                op2 = stack.pop()
                stack.append(f'({op2}{char}{op1})')

        return stack[-1]

    def prefix_to_infix(self, prefix):
        stack = []

        for char in reversed(prefix):
            if self.is_operand(char):
                stack.append(char)
            else:
                # Pop two operands and create expression
                op1 = stack.pop()
                op2 = stack.pop()
                stack.append(f'({op1}{char}{op2})')

        return stack[-1]


# Example Usage
expression = "a+[(b*{c+[d-e]})/f]*g"  # Infix expression
converter = ExpressionConverter(expression)

print("Infix Expression:", expression)
postfix = converter.infix_to_postfix()
print("Postfix Expression:", postfix)
prefix = converter.infix_to_prefix()
print("Prefix Expression:", prefix)
print("Converted back to Infix from Postfix:", converter.postfix_to_infix(postfix))
print("Converted back to Infix from Prefix:", converter.prefix_to_infix(prefix))