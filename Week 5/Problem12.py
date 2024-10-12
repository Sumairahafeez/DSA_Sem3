import time

# Declare a list
numbers = [-10, 15, -5, 20, 30, -1, 50]

# Start time measurement for removing negative numbers
startTime = time.time()
# Remove all negative numbers
numbers = [num for num in numbers if num >= 0]
endTime = time.time()
totalTime = endTime - startTime
print(f'It took {totalTime} seconds to remove negative numbers.')

# Start time measurement for finding maximum and minimum values
startTime = time.time()
max_value = max(numbers) if numbers else None
min_value = min(numbers) if numbers else None
endTime = time.time()
totalTime = endTime - startTime
print(f'It took {totalTime} seconds to find the maximum and minimum values.')

# Start time measurement for computing the average
startTime = time.time()
average = sum(numbers) / len(numbers) if numbers else None
endTime = time.time()
totalTime = endTime - startTime
print(f'It took {totalTime} seconds to compute the average.')

# Output results
print(f'Resulting List: {numbers}')
print(f'Maximum Value: {max_value}')
print(f'Minimum Value: {min_value}')
print(f'Average: {average}')
