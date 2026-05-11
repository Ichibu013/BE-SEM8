/*
Name - Ansh Shamra
BE A Computer
Roll No - 85

Problem Statement: Implement Min, Max, Sum and Average operations using Parallel Reduction.
*/

#include <iostream>
#include <vector>
#include <omp.h>
#include <random>
#include <chrono>
#include <limits>

class ParallelReduction {
private:
    std::vector<int> data;

public:
    ParallelReduction(int size) {
        // Initialize with random numbers
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 10000);

        data.resize(size);
        for(int i = 0; i < size; i++) {
            data[i] = dis(gen);
        }
    }

    void sequential_reduction() {
        auto start = std::chrono::high_resolution_clock::now();

        int min_val = std::numeric_limits<int>::max();
        int max_val = std::numeric_limits<int>::min();
        long long sum = 0;

        for(int num : data) {
            min_val = std::min(min_val, num);
            max_val = std::max(max_val, num);
            sum += num;
        }
        double avg = static_cast<double>(sum) / data.size();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Sequential Results:\n";
        std::cout << "Min: " << min_val << "\n";
        std::cout << "Max: " << max_val << "\n";
        std::cout << "Sum: " << sum << "\n";
        std::cout << "Average: " << avg << "\n";
        std::cout << "Time taken: " << duration.count() << " microseconds\n\n";
    }

    void parallel_reduction() {
        auto start = std::chrono::high_resolution_clock::now();

        int min_val = std::numeric_limits<int>::max();
        int max_val = std::numeric_limits<int>::min();
        long long sum = 0;

        #pragma omp parallel reduction(min:min_val) reduction(max:max_val) reduction(+:sum)
        {
            #pragma omp for
            for(size_t i = 0; i < data.size(); i++) {
                min_val = std::min(min_val, data[i]);
                max_val = std::max(max_val, data[i]);
                sum += data[i];
            }
        }
        double avg = static_cast<double>(sum) / data.size();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Parallel Results:\n";
        std::cout << "Min: " << min_val << "\n";
        std::cout << "Max: " << max_val << "\n";
        std::cout << "Sum: " << sum << "\n";
        std::cout << "Average: " << avg << "\n";
        std::cout << "Time taken: " << duration.count() << " microseconds\n\n";
    }
};

int main() {
    const int SIZE = 10000000;
    omp_set_num_threads(4);

    ParallelReduction pr(SIZE);
    pr.sequential_reduction();
    pr.parallel_reduction();

    return 0;
}