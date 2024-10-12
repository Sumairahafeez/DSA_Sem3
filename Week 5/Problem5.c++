#include <iostream>
#include <vector>

// Function to print the matrix
void printMatrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
}

// Function to add a row to the matrix
void addRow(std::vector<std::vector<int>>& matrix, const std::vector<int>& newRow) {
    matrix.push_back(newRow);
}

// Function to add a column to the matrix
void addColumn(std::vector<std::vector<int>>& matrix, const std::vector<int>& newColumn) {
    for (size_t i = 0; i < matrix.size(); ++i) {
        // If the newColumn size is smaller than the matrix, we fill with 0s
        if (i < newColumn.size()) {
            matrix[i].push_back(newColumn[i]);
        } else {
            matrix[i].push_back(0); // Or any default value
        }
    }
}

// Function to transpose the matrix
std::vector<std::vector<int>> transpose(const std::vector<std::vector<int>>& matrix) {
    std::vector<std::vector<int>> transposed(matrix[0].size(), std::vector<int>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

int main() {
    // Create a 2D vector (matrix)
    std::vector<std::vector<int>> matrix;

    // Adding rows to the matrix
    addRow(matrix, {1, 2, 3});
    addRow(matrix, {4, 5, 6});
    addRow(matrix, {7, 8, 9});

    std::cout << "Original Matrix:" << std::endl;
    printMatrix(matrix);

    // Adding a new column to the matrix
    addColumn(matrix, {10, 11, 12});
    std::cout << "Matrix after adding a column:" << std::endl;
    printMatrix(matrix);

    // Transposing the matrix
    std::vector<std::vector<int>> transposedMatrix = transpose(matrix);
    std::cout << "Transposed Matrix:" << std::endl;
    printMatrix(transposedMatrix);

    return 0;
}
