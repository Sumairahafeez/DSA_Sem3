from itertools import permutations

def travelling_salesman_problem(graph, start):
    # Store all vertex apart from source vertex
    vertex = []
    for i in range(len(graph)):
        if i != start:
            vertex.append(i)

    # Store minimum weight Hamiltonian Cycle
    min_path = float('inf')
    next_permutation = permutations(vertex)
    for perm in next_permutation:
        # Store current Path weight(cost)
        current_pathweight = 0

        # Compute current path weight
        k = start
        for j in perm:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][start]

        # Update minimum
        min_path = min(min_path, current_pathweight)
        
    return min_path

# Driver Code
if __name__ == "__main__":
    # Matrix representation of graph
    graph = [[0, 10, 15, 20],
             [10, 0, 35, 25],
             [15, 35, 0, 30],
             [20, 25, 30, 0]]
    start = 0
    print("Minimum cost of travelling salesman problem is", travelling_salesman_problem(graph, start))