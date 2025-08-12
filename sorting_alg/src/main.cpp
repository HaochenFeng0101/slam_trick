// main.cpp
#include "sort.hpp"
#include <iostream>

void printVector(const std::vector<int>& arr) {
    for (int val : arr) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> nums = {5, 2, 9, 1, 5, 6};

    std::cout << "Original array: ";
    printVector(nums);

    // Create a copy to sort
    std::vector<int> bubble_nums = nums;
    MySort::bubbleSort(bubble_nums);
    std::cout << "Bubble Sorted: ";
    printVector(bubble_nums);

    std::vector<int> merge_nums = nums;
    MySort::mergeSort(merge_nums);
    std::cout << "Merge Sorted: ";
    printVector(merge_nums);

    std::vector<int> heap_nums = nums;
    MySort::heapSort(heap_nums);
    std::cout << "Heap Sorted: ";
    printVector(heap_nums);

    std::vector<int> binary_nums = nums;
    MySort::binaryInsertionSort(binary_nums);
    std::cout << "Binary Insertion Sorted: ";
    printVector(binary_nums);

    return 0;
}