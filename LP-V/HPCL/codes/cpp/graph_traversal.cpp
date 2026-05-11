/*
Name - Ansh Sharma
BE A Computer
Roll No - 85

Problem Statement: Design and implement Parallel Breadth First Search and Depth First Search
based on existing algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS.
*/

#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <string>
#include <mutex>

using namespace std;

// Simple logging utility
void logInfo(const string& msg) {
    // Prevent threads from scrambling standard output
    #pragma omp critical (print_lock)
    {
        cout << "[INFO] " << msg << endl;
    }
}

class Graph {
private:
    int V;
    vector<vector<int>> adj;

    // Helper for Sequential DFS
    void dfs_recursive(int current, vector<bool>& visited) {
        visited[current] = true;
        cout << current << " ";

        for (int neighbor : adj[current]) {
            if (!visited[neighbor]) {
                dfs_recursive(neighbor, visited);
            }
        }
    }

    // Helper for Parallel DFS
    void dfs_recursive_task(int current, vector<bool>& visited) {
        // Print safely using critical section
        #pragma omp critical (print_lock)
        {
            cout << current << " ";
        }

        for (int neighbor : adj[current]) {
            bool should_explore = false;

            // Safely check and mark visited
            if (!visited[neighbor]) {
                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        should_explore = true;
                    }
                }
            }

            // If unvisited, spawn an OpenMP task to explore this branch
            if (should_explore) {
                #pragma omp task
                {
                    dfs_recursive_task(neighbor, visited);
                }
            }
        }
        // Wait for all spawned tasks (child branches) to finish before returning
        #pragma omp taskwait
    }

public:
    Graph(int vertices) {
        V = vertices;
        adj.resize(V);
    }

    void addEdge(int v, int w) {
        adj[v].push_back(w);
        adj[w].push_back(v);
    }

    // ---------------------------------------------------------
    // 1. SEQUENTIAL BFS
    // ---------------------------------------------------------
    void sequential_bfs(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "[INFO] Sequential BFS Path: ";
        while (!q.empty()) {
            int current = q.front();
            q.pop();
            cout << current << " ";

            for (int neighbor : adj[current]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        cout << endl;
    }

    // ---------------------------------------------------------
    // 2. PARALLEL BFS (Level-Synchronous Approach)
    // ---------------------------------------------------------
    void parallel_bfs(int start) {
        vector<bool> visited(V, false);
        vector<int> current_level;

        visited[start] = true;
        current_level.push_back(start);

        cout << "[INFO] Parallel BFS Path (grouped by level):" << endl;

        // Process level by level
        while (!current_level.empty()) {
            vector<int> next_level;

            // Print current level nodes
            cout << "Level: ";
            for (int node : current_level) cout << node << " ";
            cout << endl;

            // OpenMP Parallel Region
            #pragma omp parallel
            {
                // Thread-local vector prevents massive thread contention (locks)
                // when multiple threads try to push to next_level simultaneously.
                vector<int> local_next_level;

                // Distribute current level nodes among available threads
                #pragma omp for
                for (size_t i = 0; i < current_level.size(); i++) {
                    int current = current_level[i];

                    for (int neighbor : adj[current]) {
                        // Double-checked locking to safely update shared 'visited' array
                        if (!visited[neighbor]) {
                            bool should_push = false;

                            #pragma omp critical
                            {
                                if (!visited[neighbor]) {
                                    visited[neighbor] = true;
                                    should_push = true;
                                }
                            }

                            if (should_push) {
                                local_next_level.push_back(neighbor);
                            }
                        }
                    }
                }

                // Merge thread-local vectors into the global next_level safely
                #pragma omp critical
                {
                    next_level.insert(next_level.end(), local_next_level.begin(), local_next_level.end());
                }
            }

            // Move to the next depth level
            current_level = next_level;
        }
    }

    // ---------------------------------------------------------
    // 3. SEQUENTIAL DFS
    // ---------------------------------------------------------
    void sequential_dfs(int start) {
        vector<bool> visited(V, false);
        cout << "[INFO] Sequential DFS nodes discovered: ";
        dfs_recursive(start, visited);
        cout << endl;
    }

    // ---------------------------------------------------------
    // 4. PARALLEL DFS (Task-based Approach)
    // ---------------------------------------------------------
    void parallel_dfs(int start) {
        vector<bool> visited(V, false);
        visited[start] = true;

        cout << "[INFO] Parallel DFS nodes discovered: " << endl;

        // Start parallel region
        #pragma omp parallel
        {
            // Only one master thread starts the recursive tree
            #pragma omp single
            {
                dfs_recursive_task(start, visited);
            }
        }
        cout << endl;
    }
};

// ---------------------------------------------------------
// MAIN EXECUTION
// ---------------------------------------------------------
int main() {
    logInfo("Initializing Graph with 8 vertices...");
    Graph g(8);

    /* Creating the test graph:
       0 -- 1 -- 3
       |    |
       2 -- 4
       |\
       5 6 -- 7
    */
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);
    g.addEdge(6, 7);

    logInfo("Graph created successfully.");
    cout << "-------------------------------------------\n";

    g.sequential_bfs(0);
    cout << "-------------------------------------------\n";

    g.parallel_bfs(0);
    cout << "-------------------------------------------\n";

    g.sequential_dfs(0);
    cout << "-------------------------------------------\n";

    g.parallel_dfs(0);
    cout << "-------------------------------------------\n";

    return 0;
}