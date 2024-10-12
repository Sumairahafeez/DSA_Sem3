#include <iostream>
using namespace std;
struct  Node
{
    int Data;
    Node* next;
};

class StackLinkedList
{
    private:
    Node* top;
    public:
    StackLinkedList()
    {
        top = nullptr;
    }
    void Push(int value)
    {
        Node* newNode = new Node();
        newNode ->Data = value;
        newNode ->next = top;
        top = newNode;
    }
    int Pop()
    {
        if(top == nullptr)
        {
            cout<<"Stack is empty";
            return -1;
        }
        else
        {
            Node* newNode = top;
            int value = top->Data;
            top = top->next;
            delete newNode;
            return value;
        }
    }
    int Peek()
    {
        if(top == nullptr)
        {
            cout<<"Stack is empty";
            return -1;
        }
        else
        {
            return top->Data;
        }
    }
    void display() {
        if (top == 0) {
            cout << "Stack is empty!" << endl;
            return;
        }
        Node* current = top;
        while (current != nullptr) {
            cout << current->Data << " ";
            current = current->next;
        }
        cout << endl;
    }
};
bool isValidOperand(string operand)
{
    if(operand == "+" || operand == "-" || operand == "*" || operand == "/" || operand == "%")
    {
        return true;
    }
    return false;
}
int PerformOperation(int val1,int val2,string operation)
{
    if(operation == "+")
    {
        return val1+val2;
    }
    else if(operation == "-")
    {
        return val1-val2;
    }
    else if(operation == "*")
    {
        return val1*val2;
    }
    else if(operation == "/")
    {   if(val2 == 0)
    {
        cout<<"Division iwth 0 is not possible"<<endl;
        return 0;
    }
        return val1/val2;
    }
    else if(operation == "%")
    {
        if(val2 == 0)
        {
            cout<<"modulus with 0 not possible"<<endl;
            return 0;
        }
        return val1%val2;
    }
    return 0;
}
int main()
{
    int value;
    string operand="";
    StackLinkedList newStack;
    while(operand != "!")
    {
    cout<<"Enter the number: ";
    if(cin>>value)
    {
        newStack.Push(value);
        cout<<"Enter the operand(+,-,*,/,%)";
        cin>>operand;
        if(isValidOperand(operand))
        {   if(newStack.Peek()==-1)
        {
            continue;
        }
            int val1 = newStack.Pop();
            int val2 = newStack.Pop();
            int result = PerformOperation(val1,val2,operand);
            newStack.Push(result);
            cout<<result<<" is stored in stack"<<endl;
        }
        if(operand == "^")
        {
            cout<<"Top of stack is "<<newStack.Peek()<<endl;
        }
        else if(operand == "!")
        {
            cout<<"exiting..."<<endl;
        }
        else
        {
            cout<<"Invalid Operand"<<endl;
        }
    }
    else
    {
        cout<<"value is not correct"<<endl;
    }   
    }
    

}