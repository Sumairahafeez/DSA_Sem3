def printMat(A,starting_index,rows,columns):
   # Get the matrix dimensions
    maxRows = len(A)
    maxCols = len(A[0]) if maxRows > 0 else 0
    
    # Unpack the starting index
    startRow, startCol = starting_index
    result = [[0 for _ in range(startCol,maxCols)]for _ in range(startRow,maxRows)]
    # Loop through the number of rows
    for i in range(rows):
        currentRow = startRow + i
        
        # Check if the current row is within the matrix bounds
        if currentRow >= maxRows:
            break
        
        # For each row, loop through the number of columns
        for j in range(columns):
            currentCol = startCol + j
            
            # Check if the current column is within the matrix bounds
            if currentCol >= maxCols:
                break
            
            # Print the matrix element at the calculated position
            print(A[currentRow][currentCol], end=' ')
            result[i][j] =  A[currentRow][currentCol]
            
        
        # Print a newline after each row
        print()
    return result    
def MatAdd(A,B):
        # Get the rows and columns from matrix A
        maxRows = len(A)
        maxCol = len(A[0])if maxRows>0 else 0
        # initialize the resulting matrix with zeros of the same size
        result = [[0 for _ in range(maxCol)]for _ in range(maxRows)]
        # Iterate over the rows and columns to get the sum
        for i in range(maxRows):
             for j in range(maxCol):
                  result[i][j] = A[i][j]+B[i][j]
        return result   
    #Addition of Partial Matrices
def MatAddPartial(A,B,start,size):
    new_A = printMat(A,start,size,size)
    new_B = printMat(B,start,size,size)
    return (MatAdd(new_A,new_B))  
# Function to multiply matrices using loops
def MatMul(A,B):
    maxRowsA = len(A)
    maxColsA = len(A[0]) if maxRowsA>0 else 0
    maxRowsB = len(B)
    maxColsB = len(B[0]) if maxRowsB>0 else 0
    result = [[0 for _ in range(maxColsB)]for _ in range(maxRowsA)]
    if maxRowsA != maxColsB:
        return "Cannot multiply, number of columns in A must equal number of rows in B"
    
    for i in range(maxRowsA):
         for j in range(maxColsB):
              for k in range(maxColsA):
                  result[i][j] += A[i][k]*B[k][j]
    return result
# Helper function to split matrices into 4 parts
def splitMatrix(A):
    n = len(A)
    mid = n // 2
    A11 = [[A[i][j] for j in range(mid)] for i in range(mid)]
    A12 = [[A[i][j] for j in range(mid, n)] for i in range(mid)]
    A21 = [[A[i][j] for j in range(mid)] for i in range(mid, n)]
    A22 = [[A[i][j] for j in range(mid, n)] for i in range(mid, n)]
    return A11, A12, A21, A22
                 
# Function for multiplying Recursively
def MatMulRecursiver(A,B):
    n = len(A)
    if not isTwoxTwo(A) and not isTwoxTwo(B):
        return "Given Matrixes cannot be multiplied"
    # Base case for 1x1 matrices
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    # Split A and B into 4 submatrices
    A11, A12, A21, A22 = splitMatrix(A)
    B11, B12, B21, B22 = splitMatrix(B)
    
    # Recursive calls to multiply submatrices
    M1 = MatAdd(MatMulRecursiver(A11, B11), MatMulRecursiver(A12, B21))
    M2 = MatAdd(MatMulRecursiver(A11, B12), MatMulRecursiver(A12, B22))
    M3 = MatAdd(MatMulRecursiver(A21, B11), MatMulRecursiver(A22, B21))
    M4 = MatAdd(MatMulRecursiver(A21, B12), MatMulRecursiver(A22, B22))
    
    # Combine the four submatrices into the final result
    result = [[0 for _ in range(2 * len(M1[0]))] for _ in range(2 * len(M1))]
    
    for i in range(len(M1)):
        for j in range(len(M1[0])):
            result[i][j] = M1[i][j]
            result[i][j + len(M1[0])] = M2[i][j]
            result[i + len(M1)][j] = M3[i][j]
            result[i + len(M1)][j + len(M1[0])] = M4[i][j]
    
    return result
# Function to get difference of Matrices
def MatSubtract(A, B):
    rows = len(A)
    cols = len(A[0]) if rows >0 else 0
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = (A[i][j]) - (B[i][j])
    return result
# Function to test if a given matrix is 2x2
def isTwoxTwo(A):
    if len(A)%2 == 0 and len(A[0])%2 == 0:
        return True
    return False
# Recursive Strassen's algorithm
def MatMulStrassen(A, B):
    n = len(A)
    # As it can be performed on 2x2 Matrix multiples
    if not isTwoxTwo(A) and not isTwoxTwo(B):
        return "Given Matrixes cannot be multiplied"
    # Base case for 1x1 matrices
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    
    # Split matrices into 4 parts
    A11, A12, A21, A22 = splitMatrix(A)
    B11, B12, B21, B22 = splitMatrix(B)
    
    # Apply Strassen's formula
    M1 = MatMulStrassen(MatAdd(A11, A22), MatAdd(B11, B22))
    M2 = MatMulStrassen(MatAdd(A21, A22), B11)
    M3 = MatMulStrassen(A11, MatSubtract(B12, B22))
    M4 = MatMulStrassen(A22, MatSubtract(B21, B11))
    M5 = MatMulStrassen(MatAdd(A11, A12), B22)
    M6 = MatMulStrassen(MatSubtract(A21, A11), MatAdd(B11, B12))
    M7 = MatMulStrassen(MatSubtract(A12, A22), MatAdd(B21, B22))
    
    # Calculate the submatrices of the result matrix C
    C11 = MatAdd(MatSubtract(MatAdd(M1, M4), M5), M7)
    C12 = MatAdd(M3, M5)
    C21 = MatAdd(M2, M4)
    C22 = MatAdd(MatSubtract(MatAdd(M1, M3), M2), M6)
    
    # Combine the results into one matrix
    result = [[0 for _ in range(2 * len(C11[0]))] for _ in range(2 * len(C11))]
    
    for i in range(len(C11)):
        for j in range(len(C11[0])):
            result[i][j] = C11[i][j]
            result[i][j + len(C11[0])] = C12[i][j]
            result[i + len(C11)][j] = C21[i][j]
            result[i + len(C11)][j + len(C11[0])] = C22[i][j]
    
    return result      
              

#Get Input  
A = [
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7],
]
B =  [
    [1, 5, 3, 4],
    [2, 3, 4, 5],
    [3, 5, 5, 6],
    [4, 5, 8, 7],
]
starting_index = (2, 4)
rows = 2
columns = 3
# Print All the Result
printMat(A, starting_index, rows, columns) 
print(MatAdd(A,B)) 
print(MatAddPartial(A,B,starting_index,2))      
print(MatMul(A,B))
print(MatMulRecursiver(A,B))
print(MatMulStrassen(A,B))