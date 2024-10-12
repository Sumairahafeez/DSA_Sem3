import time

# Function to create an empty grid
def createGrid(rows, cols):
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    return grid

# Function to add a row to the grid
def addRow(grid):
    startTime = time.time()
    newRow = [0] * len(grid[0])  # Add a new row with zeros
    grid.append(newRow)
    endTime = time.time()
    totalTime = endTime - startTime
    print(f'It took {totalTime} seconds to add a new row.')

# Function to add a column to the grid
def addColumn(grid):
    startTime = time.time()
    for row in grid:
        row.append(0)  # Add a new column with zeros to each row
    endTime = time.time()
    totalTime = endTime - startTime
    print(f'It took {totalTime} seconds to add a new column.')

# Function to display the grid
def displayGrid(grid):
    print("Current Grid:")
    for row in grid:
        print(row)

# Function to calculate the sum of all elements in the grid
def sumGrid(grid):
    total = 0
    startTime = time.time()
    for row in grid:
        total += sum(row)
    endTime = time.time()
    totalTime = endTime - startTime
    print(f'It took {totalTime} seconds to calculate the sum.')
    return total

# Main Program
if __name__ == "__main__":
    rows = 3
    cols = 3
    myGrid = createGrid(rows, cols)  # Initialize a 3x3 grid

    displayGrid(myGrid)
    addRow(myGrid)  # Adding a row
    addColumn(myGrid)  # Adding a column
    displayGrid(myGrid)

    gridSum = sumGrid(myGrid)  # Calculate the sum of elements
    print(f'The sum of all elements in the grid is: {gridSum}')
