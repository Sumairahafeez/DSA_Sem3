#include<iostream>
using namespace std;

class QueueArray
{
    private:
        int array[100];
        int front;
        int rear;
    public:
    QueueArray()
    {
        front = -1;
        rear = -1;
    }
    bool isFull()
    {
        return (rear+1 == front);
    }
    bool isEmpty()
    {
        return (front == rear || front == -1);
    }
    void Enqueue(int x)
    {
        if(isFull())
        {
            cout<<"Queue is full!"<<endl;
        }
        else
        {
            array[rear+1] = x;
            if(rear == 100-1)
            {
                rear == 0;
            }
            else
            {
                rear = rear+1;
            }   
        }
        
    }
    int Dequeue()
    {
        if(isEmpty())
        {
            cout<<"Array is empty"<<endl;
        }
        else
        {
            int temp = array[front];
            front-=1;
            return temp;
        }
        if(front = 100-1)
        {
            front = 0;
        }
    }
};
struct Node
{
    int Data;
    Node* next;
};

class QueueLinkedList
{
    private:
        Node* rear;
        Node* front;
    public:
        QueueLinkedList()
        {
            rear = nullptr;
            front = nullptr;
        }
        bool isEmpty()
        {
            return front == nullptr;
        }
        // bool isFull()
        // {
        //     return rear->next = front
        // }
        void Enqueue(int value)
        {
            Node* newNode = new Node();
            newNode ->Data = value;
            newNode ->next = NULL;
            if(rear == nullptr)
            {
                front = rear = newNode;
            }
            else
            {
                rear ->next = newNode;
                rear = newNode;
            }
        }
        int Dequeue()
        {
            if(isEmpty())
            {
                cout<<"Queue is Empty";
            }
            else
            {
                Node* newNode = front;
                int value = front->Data;
                front = front->next;
                if(front ==nullptr)
                {
                    rear == nullptr;
                }
                delete newNode;
                return value;
            }
        }
        int Peek()
        {
            if(isEmpty())
            {
                return -1;
            }
            return front->Data;
        }
        void display() {
        if (isEmpty()) {
            cout << "Queue is empty!" << endl;
            return;
        }
        Node* current = front;
        while (current != nullptr) {
            cout << current->Data << " ";
            current = current->next;
        }
        cout << endl;
    }
};
int main() {
    QueueLinkedList queue;
    queue.Enqueue(10);
    queue.Enqueue(20);
    queue.Enqueue(30);
    cout << "Queue elements: ";
    queue.display();
    cout << "Front element: " << queue.Peek() << endl;
    cout << "Dequeued element: " << queue.Dequeue() << endl;
    cout << "Queue after dequeuing: ";
    queue.display();
    return 0;
}