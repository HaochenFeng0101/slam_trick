#include "sort.hpp"
#include <utility> // For std::swap
#include <algorithm> // Potentially for other helpers

namespace MySort {

// --- Bubble Sort ---
void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    bool swapped;
    for (int i = 0; i < n - 1; ++i) {
        swapped = false;
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }
}

// --- Merge Sort ---
// Private helper function for merge sort
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> leftArr(n1);
    std::vector<int> rightArr(n2);

    for (int i = 0; i < n1; ++i) {
        leftArr[i] = arr[left + i];
    }
    for (int j = 0; j < n2; ++j) {
        rightArr[j] = arr[mid + 1 + j];
    }

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k++] = leftArr[i++];
        } else {
            arr[k++] = rightArr[j++];
        }
    }
    while (i < n1) {
        arr[k++] = leftArr[i++];
    }
    while (j < n2) {
        arr[k++] = rightArr[j++];
    }
}

// Private recursive function for merge sort
void mergeSort_recursive(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort_recursive(arr, left, mid);
        mergeSort_recursive(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Public wrapper function
void mergeSort(std::vector<int>& arr) {
    if (!arr.empty()) {
        mergeSort_recursive(arr, 0, arr.size() - 1);
    }
}

// --- Heap Sort ---
// Private helper function for heap sort
void heapify(std::vector<int>& arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest]) {
        largest = left;
    }
    if (right < n && arr[right] > arr[largest]) {
        largest = right;
    }
    if (largest != i) {
        std::swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

// Public function
void heapSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = n / 2 - 1; i >= 0; --i) {
        heapify(arr, n, i);
    }
    for (int i = n - 1; i > 0; --i) {
        std::swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

// --- Binary Insertion Sort ---
// Private helper function for binary search
int binarySearch(const std::vector<int>& arr, int item, int low, int high) {
    if (high <= low) {
        return (item > arr[low]) ? (low + 1) : low;
    }
    int mid = low + (high - low) / 2;
    if (item == arr[mid]) {
        return mid + 1;
    }
    if (item > arr[mid]) {
        return binarySearch(arr, item, mid + 1, high);
    }
    return binarySearch(arr, item, low, mid - 1);
}

// Public function
void binaryInsertionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;

        int loc = binarySearch(arr, key, 0, j);

        while (j >= loc) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

} // namespace MySort