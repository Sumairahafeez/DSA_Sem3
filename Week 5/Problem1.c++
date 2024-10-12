#include <iostream>
#include <vector>
#include <string>

void displayVectorInfo(const std::vector<std::string>& vec) {
    std::cout << "Size: " << vec.size() << ", Capacity: " << vec.capacity() << std::endl;
}

int main() {
    std::vector<std::string> stringVector;
    int choice;

    do {
        std::cout << "\nMenu:\n";
        std::cout << "1. Add a string\n";
        std::cout << "2. Remove a string\n";
        std::cout << "3. Display vector info\n";
        std::cout << "4. Exit\n";
        std::cout << "Enter your choice (1-4): ";
        std::cin >> choice;

        switch (choice) {
            case 1: {
                std::string newString;
                std::cout << "Enter a string to add: ";
                std::cin.ignore(); // Ignore the newline character left in the buffer
                std::getline(std::cin, newString); // Use getline to read strings with spaces
                stringVector.push_back(newString);
                std::cout << "\"" << newString << "\" added to the vector." << std::endl;
                displayVectorInfo(stringVector);
                break;
            }
            case 2: {
                if (stringVector.empty()) {
                    std::cout << "The vector is empty. No strings to remove." << std::endl;
                } else {
                    std::string removeString;
                    std::cout << "Enter a string to remove: ";
                    std::cin.ignore();
                    std::getline(std::cin, removeString);

                    int it = std::find(stringVector.begin(), stringVector.end(), removeString);
                    if (it != stringVector.end()) {
                        stringVector.erase(it);
                        std::cout << "\"" << removeString << "\" removed from the vector." << std::endl;
                    } else {
                        std::cout << "\"" << removeString << "\" not found in the vector." << std::endl;
                    }
                    displayVectorInfo(stringVector);
                }
                break;
            }
            case 3:
                displayVectorInfo(stringVector);
                break;
            case 4:
                std::cout << "Exiting the program." << std::endl;
                break;
            default:
                std::cout << "Invalid choice! Please enter a number between 1 and 4." << std::endl;
        }
    } while (choice != 4);

    return 0;
}

