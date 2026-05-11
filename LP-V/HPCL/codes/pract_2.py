import random
import time
import multiprocessing as mp
from multiprocessing import Pool
import logging

# ---------------------------------------------------------
# Configure Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------------------------------------------
# 1. BUBBLE SORT IMPLEMENTATIONS
# ---------------------------------------------------------

def sequential_bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def sort_pair(pair):
    if len(pair) == 2 and pair[0] > pair[1]:
        pair[0], pair[1] = pair[1], pair[0]
    return pair

def parallel_bubble_sort(arr):
    n = len(arr)
    swapped = True
    iteration = 0

    while swapped:
        swapped = False
        iteration += 1

        # EVEN PHASE
        with Pool() as pool:
            chunks = [(arr[i:i+2]) for i in range(0, n-1, 2)]
            results = pool.map(sort_pair, chunks)

            for i, pair in enumerate(results):
                if len(pair) == 2:
                    if arr[i*2:i*2+2] != pair:
                        arr[i*2:i*2+2] = pair
                        swapped = True

        # ODD PHASE
        with Pool() as pool:
            chunks = [(arr[i:i+2]) for i in range(1, n-1, 2)]
            results = pool.map(sort_pair, chunks)

            for i, pair in enumerate(results):
                if len(pair) == 2:
                    if arr[i*2+1:i*2+3] != pair:
                        arr[i*2+1:i*2+3] = pair
                        swapped = True

    return arr

# ---------------------------------------------------------
# 2. MERGE SORT IMPLEMENTATIONS
# ---------------------------------------------------------

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

def sequential_merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = sequential_merge_sort(arr[:mid])
    right = sequential_merge_sort(arr[mid:])

    return merge(left, right)

def parallel_merge_sort(arr, depth=0):
    if len(arr) <= 1:
        return arr

    MAX_DEPTH = 1
    if depth >= MAX_DEPTH:
        return sequential_merge_sort(arr)

    mid = len(arr) // 2

    with Pool(2) as pool:
        results = pool.map(parallel_merge_sort_wrapper, [(arr[:mid], depth+1), (arr[mid:], depth+1)])
        left, right = results[0], results[1]

    return merge(left, right)

def parallel_merge_sort_wrapper(args):
    arr, depth = args
    return parallel_merge_sort(arr, depth)

# ---------------------------------------------------------
# 3. MAIN EXECUTION & BENCHMARKING
# ---------------------------------------------------------

def main():
    # Reduced size to 20 so the printed arrays are actually readable!
    SIZE = 20
    logging.info(f"Generating array of {SIZE} random integers...")
    arr = [random.randint(1, 100) for _ in range(SIZE)]

    logging.info(f"ORIGINAL ARRAY: {arr}")

    logging.info("--- Starting Bubble Sort Benchmarks ---")

    # Sequential Bubble Sort
    arr_copy = arr.copy()
    start = time.time()
    sequential_bubble_sort(arr_copy)
    end = time.time()
    logging.info(f"Sequential Bubble Sort Time: {(end - start)*1000:.2f} ms\n")

    # Parallel Bubble Sort
    arr_copy = arr.copy()
    start = time.time()
    parallel_bubble_sort(arr_copy)
    end = time.time()
    logging.info(f"Parallel Bubble Sort Time: {(end - start)*1000:.2f} ms\n")

    logging.info("--- Starting Merge Sort Benchmarks ---")

    # Sequential Merge Sort
    arr_copy = arr.copy()
    start = time.time()
    sequential_merge_sort(arr_copy)
    end = time.time()
    logging.info(f"Sequential Merge Sort Time: {(end - start)*1000:.2f} ms\n")

    # Parallel Merge Sort
    arr_copy = arr.copy()
    start = time.time()
    parallel_merge_sort(arr_copy)
    end = time.time()
    logging.info(f"Parallel Merge Sort Time: {(end - start)*1000:.2f} ms\n")

if __name__ == "__main__":
    main()