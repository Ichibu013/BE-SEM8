#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>

// Helper function for structured logging
void log(int rank, const std::string& message) {
    // Only Rank 0 handles the main logging to keep console output clean
    if (rank == 0) {
        std::cout << "[Rank " << rank << "] " << message << "\n";
    }
}

int main(int argc, char** argv) {
    int rank, size;

    // Initialize the MPI Environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ==========================================
    // 1. CONFIGURATION & INITIALIZATION
    // ==========================================
    // Array size (N) - Adjust this for different test cases in your report
    const int N = 10000000; // 10 Million elements

    if (N % size != 0) {
        if (rank == 0) {
            std::cerr << "Error: Array size N (" << N << ") must be divisible by number of processes (" << size << ")." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int local_n = N / size;
    std::vector<int> global_array;
    std::vector<int> local_array(local_n);

    double seq_time = 0.0;
    double parallel_start_time, parallel_end_time;

    // ==========================================
    // 2. DATA GENERATION & SEQUENTIAL BASELINE
    // ==========================================
    if (rank == 0) {
        std::cout << "\n=======================================================\n";
        std::cout << " MPI Parallel Quicksort Performance Evaluation\n";
        std::cout << "=======================================================\n";
        std::cout << "Array Size (N) : " << N << " elements\n";
        std::cout << "Processes (P)  : " << size << "\n";
        std::cout << "-------------------------------------------------------\n";

        log(rank, "Allocating memory and generating random data...");
        global_array.resize(N);

        // Use random_device and mt19937 for better random distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 1000000);

        for (int i = 0; i < N; ++i) {
            global_array[i] = dis(gen);
        }

        // --- Calculate Sequential Baseline (Ts) ---
        log(rank, "Calculating sequential baseline (Ts) on a copy of the data...");
        std::vector<int> sequential_copy = global_array;

        double seq_start = MPI_Wtime();
        std::sort(sequential_copy.begin(), sequential_copy.end());
        double seq_end = MPI_Wtime();

        seq_time = seq_end - seq_start;
        log(rank, "Sequential sorting completed.");
    }

    // ==========================================
    // 3. PARALLEL EXECUTION
    // ==========================================

    // Synchronize all processes before starting the parallel timer
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        log(rank, "Scattering data across all " + std::to_string(size) + " processes...");
        parallel_start_time = MPI_Wtime();
    }

    // Scatter the unsorted global array to all local arrays
    MPI_Scatter(global_array.data(), local_n, MPI_INT,
                local_array.data(), local_n, MPI_INT,
                0, MPI_COMM_WORLD);

    // Every process sorts its local sub-array using optimized C++ std::sort
    std::sort(local_array.begin(), local_array.end());

    // Prepare a buffer on the root process to receive sorted chunks
    std::vector<int> gathered_array;
    if (rank == 0) {
        gathered_array.resize(N);
        log(rank, "Gathering locally sorted chunks back to Root...");
    }

    // Gather all sorted local arrays back to the root process
    MPI_Gather(local_array.data(), local_n, MPI_INT,
               gathered_array.data(), local_n, MPI_INT,
               0, MPI_COMM_WORLD);

    // ==========================================
    // 4. MERGE & EVALUATION (Root Process Only)
    // ==========================================
    if (rank == 0) {
        log(rank, "Merging sorted chunks into final array...");

        // Initialize final array with the first sorted chunk
        std::vector<int> final_sorted_array(gathered_array.begin(), gathered_array.begin() + local_n);

        // Iteratively merge the remaining chunks
        for (int p = 1; p < size; ++p) {
            std::vector<int> temp_merged(final_sorted_array.size() + local_n);

            auto chunk_start = gathered_array.begin() + (p * local_n);
            auto chunk_end = gathered_array.begin() + ((p + 1) * local_n);

            // Use C++ std::merge for safe and efficient merging
            std::merge(final_sorted_array.begin(), final_sorted_array.end(),
                       chunk_start, chunk_end,
                       temp_merged.begin());

            final_sorted_array = std::move(temp_merged);
        }

        // Stop the parallel timer
        parallel_end_time = MPI_Wtime();
        double parallel_time = parallel_end_time - parallel_start_time;

        // Calculate Metrics
        double speedup = seq_time / parallel_time;
        double efficiency = speedup / size;

        // --- Print Final Report ---
        std::cout << "\n=======================================================\n";
        std::cout << " PERFORMANCE REPORT\n";
        std::cout << "=======================================================\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Sequential Time (Ts) : " << seq_time << " seconds\n";
        std::cout << "Parallel Time (Tp)   : " << parallel_time << " seconds\n";
        std::cout << "Speedup (S)          : " << speedup << "x\n";
        std::cout << "Efficiency (E)       : " << (efficiency * 100.0) << "%\n";
        std::cout << "=======================================================\n\n";
    }

    // Finalize the MPI Environment
    MPI_Finalize();
    return 0;
}