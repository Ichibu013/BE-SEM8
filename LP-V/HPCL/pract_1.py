'''
Name - Ansh Shamra
BE A Computer
Roll No - 85

Problem Statement: Design and implement Parallel Breadth First Search and Depth First Search based on existing
algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS.
'''
import multiprocessing as mp
from multiprocessing import Manager
import concurrent.futures
from collections import deque

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[] for _ in range(vertices)]

    def add_edge(self, v1, v2):
        self.graph[v1].append(v2)
        self.graph[v2].append(v1)

    # ---------------------------------------------------------
    # 1. SEQUENTIAL BFS
    # ---------------------------------------------------------
    def sequential_bfs(self, start):
        visited = [False] * self.V
        queue = deque([start])
        visited[start] = True

        print(f"Sequential BFS starting from vertex {start}:", end=" ")
        while queue:
            vertex = queue.popleft()
            print(vertex, end=" ")

            for neighbor in self.graph[vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        print()

    # ---------------------------------------------------------
    # 2. PARALLEL BFS (Optimized)
    # ---------------------------------------------------------
    def _get_unvisited_neighbors(self, args):
        """Worker function for BFS: Returns neighbors not in the visited set."""
        vertex, visited_set = args
        # Read-only operation on visited_set, fast and safe
        return [neighbor for neighbor in self.graph[vertex] if neighbor not in visited_set]

    def parallel_bfs(self, start):
        visited = {start} # Using a standard set is much faster than Manager().list()
        current_level = [start]

        print(f"Parallel BFS starting from vertex {start}:", end=" ")

        # ProcessPoolExecutor manages the pool more cleanly
        with concurrent.futures.ProcessPoolExecutor() as executor:
            while current_level:
                for v in current_level:
                    print(v, end=" ")

                # Pass a standard set; less IPC serialization overhead
                args = [(v, visited) for v in current_level]

                # Map processes parallelly
                results = executor.map(self._get_unvisited_neighbors, args)

                next_level = []
                for neighbors in results:
                    for n in neighbors:
                        if n not in visited:
                            visited.add(n) # Update visited ONLY in the main process
                            next_level.append(n)

                current_level = next_level
        print()

    # ---------------------------------------------------------
    # 3. PARALLEL DFS
    # ---------------------------------------------------------
    def _dfs_worker(self, args):
        """Worker function for DFS: Explores a specific branch entirely."""
        start_node, visited_dict, lock = args
        local_path = []

        def dfs_recursive(node):
            # Lock required when reading/writing to shared memory
            with lock:
                if visited_dict[node]:
                    return
                visited_dict[node] = True

            local_path.append(node)

            for neighbor in self.graph[node]:
                with lock:
                    is_visited = visited_dict[neighbor]
                if not is_visited:
                    dfs_recursive(neighbor)

        dfs_recursive(start_node)
        return local_path

    def parallel_dfs(self, start):
        print(f"Parallel DFS starting from vertex {start}:", end=" ")

        # For DFS, workers need to know what other workers have visited to avoid cycles
        manager = Manager()
        visited_dict = manager.dict({i: False for i in range(self.V)})
        lock = manager.Lock()

        visited_dict[start] = True
        print(start, end=" ")

        # Get immediate branches from the start node to assign to different workers
        branches = self.graph[start]
        args = [(branch, visited_dict, lock) for branch in branches]

        # Explore the sub-trees of the root node in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self._dfs_worker, args)

        # Print the paths found by the workers
        for path in results:
            for node in path:
                print(node, end=" ")
        print()

def main():
    g = Graph(8)

    # Test graph
    # 0 -- 1 -- 3
    # |    |
    # 2 -- 4
    # |\
    # 5 6 -- 7
    edges = [(0,1), (0,2), (1,3), (1,4), (2,5), (2,6), (6,7)]
    for v1, v2 in edges:
        g.add_edge(v1, v2)

    print("--- Graph Traversals ---")
    g.sequential_bfs(0)
    g.parallel_bfs(0)
    g.parallel_dfs(0)

if __name__ == "__main__":
    main()