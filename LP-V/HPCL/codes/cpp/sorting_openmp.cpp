/*
Name - Ansh Sharma
BE A Computer
Roll No - 85

Problem Statement: Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP.
Use existing algorithms and measure the performance of sequential and parallel algorithms.
*/

#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

// ---------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------

// Simple logging utility
void logInfo(const string& msg) {
    cout << "[INFO] " << msg << endl;
}

// Function to print the array
void printArray(const string& prefix, const vector<int>& arr) {
    cout << prefix;
    for (int num : arr) {
        cout << num << " ";
    }
    cout << "\n";
}

// Function to generate a random array
vector<int> generateRandomArray(int size) {
    vector<int> arr(size);
    // Use a random device and Mersenne Twister for good random numbers
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100); // Random numbers between 1 and 100

    for (int i = 0; i < size; ++i) {
        arr[i] = dis(gen);
    }
    return arr;
}

// ---------------------------------------------------------
// 1. BUBBLE SORT IMPLEMENTATIONS
// ---------------------------------------------------------

void sequential_bubble_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

/* * Parallel Bubble Sort (Odd-Even Transposition Sort)
 * We compare and swap even-indexed pairs, then odd-indexed pairs.
 * This ensures independent pairs are swapped without race conditions.
 */
void parallel_bubble_sort(vector<int>& arr) {
    int n = arr.size();
    // It takes exactly 'n' phases to sort the array completely in the worst case
    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            // Even Phase: Compare (0,1), (2,3), (4,5)...
            #pragma omp parallel for
            for (int i = 0; i < n - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    swap(arr[i], arr[i + 1]);
                }
            }
        } else {
            // Odd Phase: Compare (1,2), (3,4), (5,6)...
            #pragma omp parallel for
            for (int i = 1; i < n - 1; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    swap(arr[i], arr[i + 1]);
                }
            }
        }
    }
}

// ---------------------------------------------------------
// 2. MERGE SORT IMPLEMENTATIONS
// ---------------------------------------------------------

// Standard merge function to combine two sorted halves
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) { arr[k] = L[i]; i++; k++; }
    while (j < n2) { arr[k] = R[j]; j++; k++; }
}

void sequential_merge_sort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        sequential_merge_sort(arr, left, mid);
        sequential_merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

/* * Parallel Merge Sort
 * Uses OpenMP Tasks to recursively sort the left and right halves simultaneously.
 */
void parallel_merge_sort_recursive(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        // If the chunk is small enough, fallback to sequential to reduce task overhead
        if ((right - left) < 100) {
            sequential_merge_sort(arr, left, right);
        } else {
            // Spawn a task for the left half
            #pragma omp task shared(arr)
            parallel_merge_sort_recursive(arr, left, mid);

            // Spawn a task for the right half
            #pragma omp task shared(arr)
            parallel_merge_sort_recursive(arr, mid + 1, right);

            // Wait for both tasks to finish before merging
            #pragma omp taskwait
            merge(arr, left, mid, right);
        }
    }
}

// Wrapper function to start the parallel OpenMP region
void parallel_merge_sort(vector<int>& arr) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_merge_sort_recursive(arr, 0, arr.size() - 1);
        }
    }
}

// ---------------------------------------------------------
// 3. MAIN EXECUTION & BENCHMARKING
// ---------------------------------------------------------

int main() {
    // Note: Set SIZE to 20 to view printed arrays cleanly.
    // Set to 50000+ to measure accurate thread performance.
    const int SIZE = 100000;

    logInfo("Generating random array...");
    vector<int> original_arr = generateRandomArray(SIZE);

    cout << "=================================================\n";
    // printArray("Original Array: ", original_arr);
    cout << "=================================================\n\n";

    // --- SEQUENTIAL BUBBLE SORT ---
    vector<int> arr_copy = original_arr;
    logInfo("Starting Sequential Bubble Sort...");
    auto start = high_resolution_clock::now();
    sequential_bubble_sort(arr_copy);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    // printArray("Array after Sequential Bubble Sort: ", arr_copy);
    cout << "Time taken: " << duration.count() << " microseconds\n\n";

    // --- PARALLEL BUBBLE SORT ---
    arr_copy = original_arr;
    logInfo("Starting Parallel Bubble Sort...");
    start = high_resolution_clock::now();
    parallel_bubble_sort(arr_copy);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    // printArray("Array after Parallel Bubble Sort: ", arr_copy);
    cout << "Time taken: " << duration.count() << " microseconds\n\n";

    // --- SEQUENTIAL MERGE SORT ---
    arr_copy = original_arr;
    logInfo("Starting Sequential Merge Sort...");
    start = high_resolution_clock::now();
    sequential_merge_sort(arr_copy, 0, arr_copy.size() - 1);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    // printArray("Array after Sequential Merge Sort: ", arr_copy);
    cout << "Time taken: " << duration.count() << " microseconds\n\n";

    // --- PARALLEL MERGE SORT ---
    arr_copy = original_arr;
    logInfo("Starting Parallel Merge Sort...");
    start = high_resolution_clock::now();
    parallel_merge_sort(arr_copy);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    // printArray("Array after Parallel Merge Sort: ", arr_copy);
    cout << "Time taken: " << duration.count() << " microseconds\n\n";

    return 0;
}