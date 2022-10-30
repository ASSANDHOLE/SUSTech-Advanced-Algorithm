#include "library.cpp"

int main() {
#ifdef OMP_ENABLED
    std::cout << "OMP enabled" << std::endl;
#else
    std::cout << "OMP disabled" << std::endl;
#endif
    float points[] = {
            0, 0,
            2, 3,
            7, 8,
            2, 6,
            4, 4,
            3, 3,
            8, 1,
            2, 5,
            10, 0,
            3, 8
    };
    int pts_num = sizeof(points) / sizeof(float) / 2;
    int k = 2;
    float m = 2;
    int init_func = 1;
    float eps = 1e-6;
    int max_iter = 1000;
    float *centers = FuzzyCMeans2dFloat(points, pts_num, k, m, init_func, eps, max_iter);
    printf("retrun 0, 1, 2, 3: %f, %f, %f, %f", centers[0], centers[1], centers[2], centers[3]);
}