#ifndef SORT_HPP
#define SORT_HPP

#include <vector>

namespace MySort {

// Bubble Sort
// Time Complexity: O(n^2), Space Complexity: O(1)
void bubbleSort(std::vector<int>& arr);

// Merge Sort
// Time Complexity: O(n log n), Space Complexity: O(n)
void mergeSort(std::vector<int>& arr);

// Heap Sort
// Time Complexity: O(n log n), Space Complexity: O(1)
void heapSort(std::vector<int>& arr);

// Binary Insertion Sort (my interpretation of "Binary Sort")
// Time Complexity: O(n^2), Space Complexity: O(1)
void binaryInsertionSort(std::vector<int>& arr);

} // namespace MySort

#endif // SORT_HPP