class Graph:
    def __init__(self,vertices):
        self.adj = [[] for _ in range(vertices)]
    def AddEdge(self,u,v):
        self.adj[u].append(v)
        self.adj[v].append(u)
    def DepthFirstSearch(self,root,visited):
        visited[root] = True
        print(root," ")
        for x in self.adj[root]:
            if not visited[x]:
                self.DepthFirstSearch(x,visited)   
    def dfs(self):
        visited = [False]*len(self.adj)
        for i in range(len(self.adj)):
            if not visited[i]:
                self.DepthFirstSearch(i,visited)
V = 6
g = Graph(V)
edges = [(0,1),(0,2),(1,3),(1,4),(2,5)]
for u,v in edges:
    g.AddEdge(u,v)
g.dfs()                                  