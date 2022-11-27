//
// Created by anguangyan on 22-11-27.
//

#include <iostream>

#include "main.cpp"

int main() {
    uint8_t A[] = {
            1, 1, 0, 0, 0,
            1, 0, 1, 0, 0,
            0, 1, 1, 0, 0,
            0, 1, 0, 1, 0,
            0, 0, 1, 1, 0,
            0, 0, 1, 0, 1,
            0, 0, 0, 1, 1,
    };
    float w[] = {1, 2, 3, 4, 5};
    int x[5];
    double result = Hw11MipSolver(A, 5, 7, w, x);
    std::cout << "result: " << result << std::endl;
    for (int i : x) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}
