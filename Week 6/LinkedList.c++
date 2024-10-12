#include<iostream>
using namespace std;

struct Node
{
    int Data;
    Node* next;
};

class LinkedList
{   private:
        Node* Head;
    public:
        LinkedList()
        {
            Head = nullptr;
        } 
        bool isEmpty()
        {
            return Head == NULL;
        }   
        Node* insertAtHead(int value)
        {
            Node* newNode = new Node();
            newNode ->Data = value;
            newNode ->next = Head;
            Head = newNode;
            return Head;
        }
        Node* insertAtEnd(int value)
        {
            Node* newNode = new Node();
            newNode ->Data = value;
            newNode ->next = NULL;
            if(isEmpty())
            {
                Head = newNode;
                return Head;
            }
            else
            {
                Node* current = Head;
                while(current ->next != NULL)
                {
                    current = current ->next;
                }
                current->next = newNode;
                return current;
            }
        }
        Node* insertNode(int index, int value)
        {
            if(index<0)
            {
                return NULL;
            }
            else if(index == 0)
            {
                insertAtHead(value);
            }
            else
            {   Node* newNode = new Node();
                newNode ->Data = value;
                Node* current = Head;
                for(int i = 0; i<index-1 && current!= NULL; i++)
                {
                    current = current->next;
                }
                if(current != NULL)
                {   
                    newNode ->next = current->next;
                    current ->next = newNode;
                }
                return current;
            }
        }
        bool FindNode(int x)
        {
            Node* current = Head;
            while (current != NULL)
            {
                if(current ->Data == x)
                {
                    return true;
                }
                current = current->next;
            }
            return false;
            
        }
        bool deleteNode(int x)
        {   
            if(Head == NULL)
            {
                return true;
            }
            bool done = false;
            Node* current = Head;
            Node* previous = NULL;
            while (current != NULL && current ->Data == x)
            {
               Node* temp = current;
               Head = current->next;
               delete temp;
               current = Head;
               done = true;
            }
            while(current != NULL)
            {
                if(current -> Data == x)
                {
                    previous ->next = current -> next;
                    delete current;
                    current = previous->next;
                    done = true;
                }
                else
                {
                    previous = current;
                    current = current ->next;
                }
            }
            return done;
        }
        bool deleteFromStart(int x)
        {
            if(Head == NULL)
            {
                return false;
            }
            Node* current = Head;
            Head = Head ->next;
            delete current;
            return true;
        }
        bool DeleteFromEnd(int x)
        {
            if(isEmpty()) return true;
            Node* current = Head;
            while(current->next->next != NULL)
            {
                current = current ->next;
            }
            delete current -> next;
            current ->next = NULL;
            return true;
        }
        void displayList()
        {
            Node* current = Head;
            while(current -> next != NULL)
            {
                cout<<current->Data<<"=>";
                current = current ->next;
            }
            cout<<"NULL";
        }
        Node* reverseList()
        {
            Node* previous = NULL;
            Node* current = Head;
            Node* next = NULL;
            // First deal with end
            while (current != NULL)
            {
               next = current->next;
               current->next = previous;
               previous = current;    
            } 
            Head = previous;
            return Head;

        }
        Node* sortList(Node* list)
        {
            if(list == NULL || list->next == NULL)
            {
                return list;
            }   
            Node* sortedList = NULL;
            Node* current = list;
            while (current != NULL)
            {
                Node* newNode = current->next;
                if(sortedList == NULL || sortedList->Data == current ->Data)
                {
                    current->next = sortedList;
                    sortedList = current;
                }
                else
                {
                    Node* temp = sortedList;
                    while(temp->next != NULL && temp->next->next->Data == current->Data)
                    {
                        temp = temp->next;
                    }
                    current ->next = temp->next;
                    temp->next = current;
                }
                current = newNode;
            }
            return sortedList;
        }
        Node* removeDuplicates(Node* list)
        {
        if (list == NULL)
        { return list;
        }
        Node* current = list;
        while (current != NULL && current->next != NULL) {
            if (current->Data == current->next->Data) {
                Node* temp = current->next;
                current->next = current->next->next;
                delete temp;
            } 
            else {
                current = current->next;
            }
        }
        return list;
        }
        Node* mergeLists(Node* list1, Node* list2) {
        if (list1 == NULL) 
        {
            return list2;
        }    
        if (list2 == NULL) 
        {
            return list1;
        }
        if (list1->Data < list2->Data) {
            list1->next = mergeLists(list1->next, list2);
            return list1;
        } else {
            list2->next = mergeLists(list1, list2->next);
            return list2;
        }
        }
        Node* interestLists(Node* list1, Node* list2)
        {
        Node* result = NULL;
        while (list1 != NULL && list2 != NULL) {
            if (list1->Data < list2->Data) {
                list1 = list1->next;
            } else if (list2->Data < list1->Data) {
                list2 = list2->next;
            } else {
                insertAtEnd(list1->Data);
                list1 = list1->next;
                list2 = list2->next;
            }
        }
        return result;
    }
};

int main()
{
    LinkedList list;
    list.insertNode(2,4);
    list.insertAtEnd(3);
    list.insertAtHead(2);
}