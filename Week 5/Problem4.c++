#include <iostream>
#include <vector>
#include <algorithm> // for std::sort and std::unique

// Function to print the elements of the vector
void printVector(const std::vector<int>& vec) {
    for (int num : vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

// Function to remove duplicates from the vector
std::vector<int> removeDuplicates(const std::vector<int>& vec) {
    std::vector<int> uniqueVec = vec; // Copy the original vector
    std::sort(uniqueVec.begin(), uniqueVec.end()); // Sort the vector
    uniqueVec.erase(std::unique(uniqueVec.begin(), uniqueVec.end()), uniqueVec.end()); // Remove duplicates
    return uniqueVec;
}

int main() {
    // Initialize a vector with some integers
    std::vector<int> myVector = {4, 2, 7, 2, 9, 4, 1, 3, 7, 5};

    // Print the original vector
    std::cout << "Original vector: ";
    printVector(myVector);

    // (a) Reverse the elements
    std::reverse(myVector.begin(), myVector.end());
    std::cout << "Reversed vector: ";
    printVector(myVector);

    // (b) Sort the elements in ascending order
    std::sort(myVector.begin(), myVector.end());
    std::cout << "Sorted vector: ";
    printVector(myVector);

    // (c) Remove duplicates
    myVector = removeDuplicates(myVector);
    std::cout << "Vector after removing duplicates: ";
    printVector(myVector);

    return 0;
}
