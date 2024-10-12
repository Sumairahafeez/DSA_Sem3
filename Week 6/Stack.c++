#include<iostream>
using namespace std;
class stackArray
{
    private:
        int array [100];
        int top;
       
    public:
        stackArray()
        {
            top = -1;
        } 
        void Push(int x)
        {
            if (top == 100)
            {
                cout<<"Stack overflow";
            }
            else
            {
                array[top+1] = x;
                top = top+1;
            }
            
        }
        int Pop()
        {
            if(top == 0)
            {
                cout<<"Stack empty";
                return -1;
            }
            else
            {
                int temp = array[top];
                top = top-1;
                return temp;
            }
        } 
        int Peek()
        {
            return array[top];
        }
        void display()
        {
        if (top == 0) 
        {
            cout << "Stack is empty!" << endl;
            return;
        }
        for (int i = top; i >= 0; i--) 
        {
            cout << array[i] << " ";
        }
        cout << endl;
        }   
};
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
        if (top == nullptr) {
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
int main() {
    StackLinkedList stack;
    stack.Push(5);
    stack.Push(8);
    stack.Push(10);
    cout << "Stack elements: ";
    stack.display();

    cout << "Top element: " << stack.Peek() << endl;

    cout << "Popped element: " << stack.Pop() << endl;
    cout << "Stack after popping: ";
    stack.display();
    stackArray stack2;

    stack2.Push(10);
    stack2.Push(20);
    stack2.Push(30);
    cout << "Stack elements: ";
    stack2.display();

    cout << "Top element: " << stack2.Peek() << endl;
    cout << "Popped element: " << stack2.Pop() << endl;
    cout << "Stack after popping: ";
    stack.display();
    return 0;
}

