from typing import List

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

class ASTHandler:
    """
    Class to convert equations to semantic parse tree, prefix to suffix etc.
    """
    def __init__(self):
        pass

    # Convert prefix expression to suffix expression
    def prefix2suffix(self, prefix: List[str]) -> List[str]:
        """
        Example:
        prefix = ['+', '1', '2']
        suffix = ['1', '2', '+']
        """
        stack = []
        for i in range(len(prefix)-1, -1, -1):
            if prefix[i] in ['+', '-', '*', '/']:
                op1 = stack.pop()
                op2 = stack.pop()
                stack.append(op1 + " " + op2 + " " + prefix[i])
            else:
                stack.append(prefix[i])
        return stack.pop()

    # Convert prefix expression to infix expression
    def prefix2infix(self, prefix: List[str]) -> List[str]:
        """Example: 
        prefix = ['+', '1', '2']
        infix = ['(', '1', '+', '2', ')']"""
        stack = []
        for i in range(len(prefix)-1, -1, -1):
            if prefix[i] in ['+', '-', '*', '/']:
                op1 = stack.pop()
                op2 = stack.pop()
                stack.append('(' +" "+ op1 +" "+ prefix[i] +" "+ op2 +" "+ ')')
            else:
                stack.append(prefix[i])
        return stack.pop()

    # Convert suffix expression to prefix expression
    def suffix2prefix(self, suffix: List[str]) -> List[str]:
        """Example
        suffix = ['1', '2', '+']
        prefix = ['+', '1', '2']
        
        suffix = ['1', '2', '+', '3', '*']
        prefix = ['*', '+', '1', '2', '3']
        """
        stack = []
        for i in range(len(suffix)):
            if suffix[i] in ['+', '-', '*', '/']:
                op1 = stack.pop()
                op2 = stack.pop()
                stack.append(suffix[i] +" "+ op2 +" "+ op1)
            else:
                stack.append(suffix[i])
        return stack.pop()

    # Convert suffix expression to infix expression
    def suffix2infix(self, suffix: List[str]) -> List[str]:
        stack = []
        for i in range(len(suffix)):
            if suffix[i] in ['+', '-', '*', '/']:
                op1 = stack.pop()
                op2 = stack.pop()
                stack.append('(' +" "+ op2 +" "+ suffix[i] +" "+ op1 +" "+ ')')
            else:
                stack.append(suffix[i])
        return stack.pop()

    def isOperator(self, c):
        return not c.isalpha() and not c.isdigit() and not has_numbers(c)

    def getPriority(self, c):
        if c == '-' or c == '+':
            return 1
        elif c == '*' or c == '/':
            return 2
        elif c == '^':
            return 3
        return 0

    def infix2suffix(self, infix):
        infix = ['('] + infix + [')']
        l = len(infix)
        char_stack = []
        output = []
        for i in range(l):
            if infix[i].isalpha() or infix[i].isdigit() or has_numbers(infix[i]):
                output.append(infix[i])
            elif infix[i] == '(':
                char_stack.append('(')
            elif infix[i] == ')':
                while char_stack[-1] != '(':
                    output.append(char_stack[-1])
                    char_stack.pop()
                char_stack.pop()
            else:
                if self.isOperator(char_stack[-1]):
                    if infix[i] == '^':
                        while self.getPriority(infix[i]) <= self.getPriority(char_stack[-1]):
                            output.append(char_stack[-1])
                            char_stack.pop()
                    else:
                        while self.getPriority(infix[i]) < self.getPriority(char_stack[-1]):
                            output.append(char_stack[-1])
                            char_stack.pop()
                    char_stack.append(infix[i])
        while char_stack:
            output.append(char_stack[-1])
            char_stack.pop()
        return " ".join(output)


    def infix2prefix(self, infix):
        print(f"Input is {infix}")
        infix.reverse()
        for i in range(len(infix)):
            if infix[i] == '(':
                infix[i] = ')'
            elif infix[i] == ')':
                infix[i] = '('
        prefix = self.infix2suffix(infix)
        prefix = prefix.split()
        prefix.reverse()
        return ' '.join(prefix)

    # Execute suffix expression
    def suffix2result(self, suffix: List[str]) -> float:
        stack = []
        for i in range(len(suffix)):
            if suffix[i] in ['+', '-', '*', '/']:
                op1 = stack.pop()
                op2 = stack.pop()
                if suffix[i] == '+':
                    stack.append(op2 + op1)
                elif suffix[i] == '-':
                    stack.append(op2 - op1)
                elif suffix[i] == '*':
                    stack.append(op2 * op1)
                elif suffix[i] == '/':
                    stack.append(op2 / op1)
            else:
                stack.append(float(suffix[i]))
        return stack.pop()

    def replace_nums(self, pattern, operands):
        """
        pattern = "+ - number2 number1 number0"
        operands = [5,3,2]
        """
        for i in range(len(operands)):
            pattern = pattern.replace("number"+str(i),str(operands[i]))
        return pattern

    def decode_preds(self, preds):
        preds = preds.split()
        return preds

if __name__ == '__main__':
    pass