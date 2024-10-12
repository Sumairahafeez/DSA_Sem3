#include <iostream>
using namespace std;

struct Node {
    int Data;
    Node* next;
    Node* previous;
};

class DoublyLinkedList {
private:
    Node* Head;

public:
    DoublyLinkedList() {
        Head = nullptr;
    }

    bool isEmpty() {
        return Head == nullptr;
    }

    Node* insertAtHead(int value) {
        Node* newNode = new Node();
        newNode->Data = value;
        newNode->next = Head;
        newNode->previous = nullptr;

        if (Head != nullptr) {
            Head->previous = newNode;
        }
        Head = newNode;
        return Head;
    }

    Node* insertAtEnd(int value) {
        Node* newNode = new Node();
        newNode->Data = value;
        newNode->next = nullptr;

        if (isEmpty()) {
            newNode->previous = nullptr;
            Head = newNode;
            return Head;
        }

        Node* current = Head;
        while (current->next != nullptr) {
            current = current->next;
        }

        current->next = newNode;
        newNode->previous = current;
        return newNode;
    }

    Node* insertNode(int index, int value) {
        if (index < 0) return nullptr;
        if (index == 0) return insertAtHead(value);

        Node* newNode = new Node();
        newNode->Data = value;

        Node* current = Head;
        for (int i = 0; i < index - 1 && current != nullptr; i++) {
            current = current->next;
        }

        if (current == nullptr) return nullptr;

        newNode->next = current->next;
        if (current->next != nullptr) {
            current->next->previous = newNode;
        }
        current->next = newNode;
        newNode->previous = current;

        return newNode;
    }

    bool deleteNode(int value) {
        if (Head == nullptr) return false;

        Node* current = Head;
        while (current != nullptr && current->Data != value) {
            current = current->next;
        }

        if (current == nullptr) return false;

        if (current->previous != nullptr) {
            current->previous->next = current->next;
        } else {
            Head = current->next;
        }

        if (current->next != nullptr) {
            current->next->previous = current->previous;
        }

        delete current;
        return true;
    }

    void displayList() {
        Node* current = Head;
        while (current != nullptr) {
            cout << current->Data << " <=> ";
            current = current->next;
        }
        cout << "NULL" << endl;
    }

    Node* reverseList() {
        Node* current = Head;
        Node* temp = nullptr;

        while (current != nullptr) {
            temp = current->previous;
            current->previous = current->next;
            current->next = temp;
            current = current->previous;
        }

        if (temp != nullptr) {
            Head = temp->previous;
        }
        return Head;
    }
};

int main() {
    DoublyLinkedList list;

    list.insertAtHead(5);
    list.insertAtEnd(10);
    list.insertAtEnd(15);
    list.insertNode(1, 7);

    cout << "Original List: ";
    list.displayList();

    list.reverseList();
    cout << "Reversed List: ";
    list.displayList();

    list.deleteNode(7);
    cout << "After deleting 7: ";
    list.displayList();

    return 0;
}
