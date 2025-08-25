//shared ptr fundemental
#include <iostream>
#include <memory> // For std::shared_ptr
#include <vector>




int main() {
    // Create a shared pointer to an integer
    std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
    std::cout << "Value pointed to by ptr1: " << *ptr1 << std::endl;

    // Create another shared pointer that shares ownership of the same integer
    std::shared_ptr<int> ptr2 = ptr1;
    std::cout << "Value pointed to by ptr2: " << *ptr2 << std::endl;

    // Check the use count (number of shared pointers pointing to the same object)
    std::cout << "Use count of ptr1: " << ptr1.use_count() << std::endl;
    std::cout << "Use count of ptr2: " << ptr2.use_count() << std::endl;

    // Reset one of the pointers
    ptr1.reset();
    std::cout << "After resetting ptr1, use count of ptr2: " << ptr2.use_count() << std::endl;

    // Check if ptr1 is null
    if (!ptr1) {
        std::cout << "ptr1 is now null." << std::endl;
    }


    //unique_ptr example
    std::unique_ptr<int> uniquePtr = std::make_unique<int>(100);
    std::cout << "Value pointed to by uniquePtr: " << *uniquePtr << std::endl;

    //weak_ptr example
    std::shared_ptr<int> sharedPtr = std::make_shared<int>(200);
    std::weak_ptr<int> weakPtr = sharedPtr;

    std::cout << "Value pointed to by sharedPtr: " << *sharedPtr << std::endl;

    if (auto lockedPtr = weakPtr.lock()) {
        std::cout << "Value pointed to by weakPtr (after lock): " << *lockedPtr << std::endl;
    } else {
        std::cout << "weakPtr is expired." << std::endl;
    }

    sharedPtr.reset(); // Reset sharedPtr, weakPtr should now be expired

    if (auto lockedPtr = weakPtr.lock()) {
        std::cout << "Value pointed to by weakPtr (after sharedPtr reset): " << *lockedPtr << std::endl;
    } else {
        std::cout << "weakPtr is expired." << std::endl;
    }

    return 0;
}
