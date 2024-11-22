from collections import deque

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def bfs(root):
    if not root:
        return []

    queue = deque([root])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node.value)

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return result
def breadthFirstSearch(adj,root,visited):
    q = deque()
    visited[root] = True
    q.append(root)
    while q:
        curr = q.popleft()
        print(curr," ")
        for x in adj[curr]:
            if visited[x] == False:
                visited[x] == True
                q.append(x)
def AddEdge(adj,u,v):
    adj[u].append(v)
    adj[v].append(u)   
def BFSDisconected(adj):
    visited = [False]*len(adj)                 
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

print(bfs(root))  # Output: [1, 2, 3, 4, 5]