#include <iostream>
using namespace std;

class stackArray {
private:
    string array[100];
    int top;

public:
    stackArray() {
        top = -1;
    }

    bool isEmpty() {
        return top == -1;
    }

    void Push(string x) {
        if (top >= 99) {
            cout << "Stack overflow" << endl;
        } else {
            array[++top] = x;
        }
    }

    string Pop() {
        if (isEmpty()) {
            cout << "Stack empty" << endl;
            return "";
        } else {
            return array[top--];
        }
    }

    string Peek() {
        if (isEmpty()) {
            cout << "Stack is empty" << endl;
            return "";
        }
        return array[top];
    }

    void display() {
        if (isEmpty()) {
            cout << "Stack is empty!" << endl;
            return;
        }
        for (int i = top; i >= 0; i--) {
            cout << array[i] << " ";
        }
        cout << endl;
    }
};

int main() {
    string Message = "I am from University of Engineering and Technology";
    stackArray newStack;

    string word = "";
    for (int i = 0; i < Message.size(); i++) {
        if (Message[i] != ' ') {
            word += Message[i];
        } else {
            if (!word.empty()) {
                newStack.Push(word);
                word = "";
            }
        }
    }
    if (!word.empty()) {
        newStack.Push(word);
    }

    while (!newStack.isEmpty()) {
        cout << newStack.Pop() << " ";
    }
    cout << endl;

    return 0;
}
