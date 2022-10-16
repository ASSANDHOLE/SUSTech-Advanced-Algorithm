#include <iostream>

bool next_combination(int *idx, int n, int k) {
    int i = k - 1;
    ++idx[i];
    while ((i > 0) && (idx[i] >= n - k + 1 + i)) {
        --i;
        ++idx[i];
    }
    if (idx[0] > n - k) {
        return false;
    }
    for (i = i + 1; i < k; ++i) {
        idx[i] = idx[i - 1] + 1;
    }
    return true;
}

int main() {
    int n = 7;
    int k = 3;
    int *idx = new int[k];
    for (int i = 0; i < k; ++i) {
        idx[i] = i;
    }
    do {
        for (int i = 0; i < k; ++i) {
            std::cout << idx[i] << " ";
        }
        std::cout << std::endl;
    } while (next_combination(idx, n, k));
}