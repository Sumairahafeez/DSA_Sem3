#include <iostream>
#include <stdexcept>

template <typename T>
class ArrayList {
private:
    T* array;         // Pointer to the dynamically allocated array
    int size;         // Current number of elements
    int capacity;     // Current capacity of the array

    // Function to resize the array
    void resize() {
        // Calculate new capacity (1.5 times the current capacity)
        int newCapacity = static_cast<int>(capacity * 1.5);
        T* newArray = new T[newCapacity]; // Create new array with the new capacity

        // Copy old elements to the new array
        for (int i = 0; i < size; i++) {
            newArray[i] = array[i];
        }

        delete[] array; // Free the old array
        array = newArray; // Point to the new array
        capacity = newCapacity; // Update capacity
    }

public:
    // Constructor
    ArrayList() : size(0), capacity(2) { // Initial capacity of 2
        array = new T[capacity]; // Allocate initial capacity
    }

    // Copy constructor
    ArrayList(const ArrayList& other) : size(other.size), capacity(other.capacity) {
        array = new T[capacity];
        for (int i = 0; i < size; i++) {
            array[i] = other.array[i];
        }
    }

    // Overloaded subscript operator for const objects
    T operator[](int index) const {
        if (index < 0 || index >= size) {
            throw std::out_of_range("Index out of range");
        }
        return array[index];
    }

    // Overloaded subscript operator for non-const objects
    T& operator[](int index) {
        if (index < 0 || index >= size) {
            throw std::out_of_range("Index out of range");
        }
        return array[index];
    }

    // Overloaded output stream operator
    friend std::ostream& operator<<(std::ostream& out, const ArrayList& other) {
        out << "[";
        for (int i = 0; i < other.size; i++) {
            out << other.array[i];
            if (i < other.size - 1) {
                out << ", ";
            }
        }
        out << "]";
        return out;
    }

    // Destructor
    ~ArrayList() {
        delete[] array; // Free allocated memory
    }

    // PushBack function to add new elements
    void PushBack(T value) {
        if (size == capacity) {
            resize(); // Resize the array if capacity is reached
        }
        array[size++] = value; // Add the new value and increment the size
    }

    // Get current size
    int getSize() const {
        return size;
    }

    // Get current capacity
    int getCapacity() const {
        return capacity;
    }
};

int main() {
    ArrayList<int> list;

    // Adding elements to the ArrayList
    for (int i = 1; i <= 10; i++) {
        list.PushBack(i);
        std::cout << "Added " << i << ", ArrayList: " << list << ", Size: " << list.getSize() << ", Capacity: " << list.getCapacity() << std::endl;
    }

    // Accessing elements
    std::cout << "Element at index 5: " << list[5] << std::endl;

    return 0;
}
