#include <iostream>
#include <vector>

int main() {
    // Declare a vector and initialize it with some integers
    std::vector<int> myVector = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    // Prompt the user to enter an integer to search for
    int target;
    std::cout << "Enter an integer to search for: ";
    std::cin >> target;

    // Initialize a flag to indicate if the integer is found
    bool found = false;

    // Search for the integer in the vector
    for (size_t i = 0; i < myVector.size(); ++i) {
        if (myVector[i] == target) {
            std::cout << "Integer " << target << " found at index: " << i << std::endl;
            found = true;  // Set the flag to true if found
            break;  // Exit the loop since we've found the target
        }
    }

    // If the integer is not found, indicate that it's not present
    if (!found) {
        std::cout << "Integer " << target << " is not present in the vector." << std::endl;
    }

    return 0;
}
