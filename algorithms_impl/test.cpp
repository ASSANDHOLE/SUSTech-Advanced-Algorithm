#include "library.cpp"

template <typename T>
void PrintVector(const std::vector<T>& v) {
    for (const auto & i : v) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

int main() {
#ifdef OMP_ENABLED
    std::cout << "OMP enabled" << std::endl;
#else
    std::cout << "OMP disabled" << std::endl;
#endif
    int subsets[] = {0, 1, 2, 0, 3, 1, 2};
    int subset_len[] = {2, 1, 2, 2};
    int subset_len_len = sizeof(subset_len) / sizeof(subset_len[0]);
    int *order = new int[subset_len_len];
    int weight[] = {4, 4, 8, 6};
    auto ret = SetCoverInt(subsets, subset_len, weight, order, subset_len_len);
    PrintVector(std::vector<int>(ret, ret + 2));
    PrintVector(std::vector<int>(order, order + subset_len_len));
}