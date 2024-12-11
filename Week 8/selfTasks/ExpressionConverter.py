class ExpansionConverter:
    precedence = {'+':1,'-':1,'*':2,'/':2,'^':3}
    def __init__(self,expression):
        self.expression = expression
    def is_operator(self,c):
        return c in ['+','-','*','/','^']
    def is_operand(self,c):
        return c.isalnum()
    def infix_to_postfix(self):
        stack = []
        postfix = ''
        for c in self.expression:
            if self.is_operand(c):
                postfix += c
            elif c == '(' or c == '[' or c == '{':
                stack.append('(')
            elif c == ')' or c == ']' or c == '}':
                while stack and stack[-1] != '([{':
                    postfix += stack.pop()
                stack.pop()
            else:
                while stack and stack[-1] != '(' and self.precedence[c] <= self.precedence[stack[-1]]:
                    postfix += stack.pop()
                stack.append(c)
        while stack:
            postfix += stack.pop()
        return postfix
    def infix_to_prefix(self):
        self.expression = self.expression[::-1]
        self.expression = self.expression.replace('(','*').replace(')','(').replace('*',')')
        self.expression = self.expression.replace('[','*').replace(']','[').replace('*',']')
        self.expression = self.expression.replace('{','*').replace('}','{').replace('*','}')
        return self.get_postfix()[::-1]
    def postfix_to_infix(self):
        stack = []
        for c in self.expression:
            if self.is_operand(c):
                stack.append(c)
            else:
                op1 = stack.pop()
                op2 = stack.pop()
                stack.append('('+op2+c+op1+')')
        return stack.pop()
    def prefix_to_infix(self):
        stack = []
        for c in self.expression[::-1]:
            if self.is_operand(c):
                stack.append(c)
            else:
                op1 = stack.pop()
                op2 = stack.pop()
                stack.append('('+op1+c+op2+')')
        return stack.pop()

expression = "{a+[(b*{c+[d-e]})/f]}*g"  # Infix expression
converter = ExpansionConverter(expression)
print("Infix Expression:", expression)
postfix = converter.infix_to_postfix()
print("Postfix Expression:", postfix)
prefix = converter.infix_to_prefix()
print("Prefix Expression:", prefix)
print("Converted back to Infix from Postfix:", converter.postfix_to_infix(postfix))
print("Converted back to Infix from Prefix:", converter.prefix_to_infix(prefix))

