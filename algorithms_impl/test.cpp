#include "library.cpp"

template <typename T>
void PrintVector(const std::vector<T>& v) {
    for (const auto & i : v) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

int main() {

    int v_n = 10;
    int edges[] = {
            0, 1,
            0, 2,
            0, 3,
            3, 9,
            0, 4,
            5, 0,
            6, 0,
            7, 0,
            8, 0,
    };
    int edge_n = sizeof(edges) / sizeof(int) / 2;
    int starts[] = {
            5, 6, 7, 8
    };
    int ends[] = {
            1, 2, 9, 4
    };
    int start_n = sizeof(starts) / sizeof(int);
    int *ret = DisjointPathProblemC1(v_n, edges, edge_n, starts, ends, start_n);
    int len = 2 + 4 + 4;
    len += ret[0] * 2;
    len += ret[1] * 2;
    PrintVector(std::vector<int>(ret, ret + len));
}