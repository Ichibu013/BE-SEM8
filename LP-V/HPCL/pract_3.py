'''
Name - Ansh Shamra
BE A Computer
Roll No - 85

Problem Statement: Implement Min, Max, Sum and Average operations using Parallel Reduction.
'''

import random
import time
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np

class ParallelReduction:
    def __init__(self, size):
        self.data = [random.randint(1, 10000) for _ in range(size)]

    def sequential_reduction(self):
        start_time = time.time()

        min_val = min(self.data)
        max_val = max(self.data)
        sum_val = sum(self.data)
        avg_val = sum_val / len(self.data)

        end_time = time.time()
        duration = (end_time - start_time) * 1_000_000  # Convert to microseconds

        print("Sequential Results:")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")
        print(f"Sum: {sum_val}")
        print(f"Average: {avg_val}")
        print(f"Time taken: {duration:.2f} microseconds\n")

    def chunk_reduction(self, chunk):
        return min(chunk), max(chunk), sum(chunk)

    def parallel_reduction(self):
        start_time = time.time()

        # Split data into chunks for parallel processing
        num_processes = mp.cpu_count()
        chunk_size = len(self.data) // num_processes
        chunks = [self.data[i:i + chunk_size] for i in range(0, len(self.data), chunk_size)]

        # Process chunks in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(self.chunk_reduction, chunks)

        # Combine results
        min_val = min(result[0] for result in results)
        max_val = max(result[1] for result in results)
        sum_val = sum(result[2] for result in results)
        avg_val = sum_val / len(self.data)

        end_time = time.time()
        duration = (end_time - start_time) * 1_000_000  # Convert to microseconds

        print("Parallel Results:")
        print(f"Min: {min_val}")
        print(f"Max: {max_val}")
        print(f"Sum: {sum_val}")
        print(f"Average: {avg_val}")
        print(f"Time taken: {duration:.2f} microseconds\n")

def main():
    SIZE = 10_000_000
    pr = ParallelReduction(SIZE)

    pr.sequential_reduction()
    pr.parallel_reduction()

if __name__ == "__main__":
    main()