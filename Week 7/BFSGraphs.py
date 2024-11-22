from collections import deque
def AddEdge(adj,u,v):
    adj[u].append(v)
    adj[v].append(u)
def breadthFirstSearch(Adj,root,visited):
    q = deque()    
    visited[root]= True
    q.append(root)
    while q:
        current = q.popleft()
        print(current," ")
        for x in Adj[current]:
            if visited[x] == False:
                visited[x] = True
                q.append(x)
def BFSDisconnected(adj):
    visited = [False]*len(adj)
    for i in range(len(adj)):
        if visited[i] == False:
            
            breadthFirstSearch(adj,i,visited)
def main():
    V = 6
    adj = [[] for _ in range(V)]
    AddEdge(adj,0,1)
    AddEdge(adj,0,2)
    AddEdge(adj,1,3)
    AddEdge(adj,1,4)
    AddEdge(adj,2,5)
    BFSDisconnected(adj) 
main()                   