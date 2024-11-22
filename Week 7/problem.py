from collections import deque,defaultdict
def bfsShortestPath(graph,start,goal):
    queue = deque([(start, [start])])
    visited = set()
    while queue:
        current, path = queue.popleft()
        visited.add(current)
        if current == goal:
            return path
        for neighbour in graph[current]:
            if neighbour not in visited:
                queue.append((neighbour, path + [neighbour]))
def dfsExists(graph, start, goal, path=None):
    if path is None:
        path = [start]
    if start == goal:
        return True
    for neighbour in graph[start]:
        if neighbour not in path:
            if dfsExists(graph, neighbour, goal, path + [neighbour]):
                return True
    return False
def rescue_mission():
    n = int(input('Enter number of nodes: '))
    graph = defaultdict(list)
    for _ in range(n):
        node, *neighbours = input().split()
        graph[node].extend(neighbours)
    start,person = map(int,input('Enter start and end positions').split()) 
    hq = int(input('Enter number of hqs'))
    shortestPath = bfsShortestPath(graph,start,person)
    directPath = dfsExists(graph,start,person)
    if shortestPath:
        print(f'Shortest path: {shortestPath}')
    else:
        print('No path found')
    if directPath:
        print('Direct path exists')
    print('total clearings:',hq)
rescue_mission()                   
