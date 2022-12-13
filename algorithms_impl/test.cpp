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
            1, 2,
            2, 3,
            4, 0,
            5, 1,
            6, 2,
            3, 7
    };
    int edge_n = sizeof(edges) / sizeof(int) / 2;
    int starts[] = {
            4, 5, 6, 0
    };
    int ends[] = {
            1, 2, 3, 7
    };
    int start_n = sizeof(starts) / sizeof(int);
    int *ret = DisjointPathProblemCn(v_n, edges, edge_n, starts, ends, start_n, 2);
    int len = 2 + 4 + 4;
    len += ret[0] * 2;
    len += ret[1] * 2;
    PrintVector(std::vector<int>(ret, ret + len));
}