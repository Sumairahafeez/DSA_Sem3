#include <iostream>
#include <vector>

int main() {
    // Declare a vector of integers
    std::vector<int> myVector;

    // Insert 100 integers into the vector
    for (int i = 0; i < 100; ++i) {
        myVector.push_back(i);  // Insert the integer into the vector

        // Print the size and capacity of the vector
        std::cout << "Inserted: " << i << " | Size: " << myVector.size() 
                  << " | Capacity: " << myVector.capacity() << std::endl;
    }

    return 0;
}
