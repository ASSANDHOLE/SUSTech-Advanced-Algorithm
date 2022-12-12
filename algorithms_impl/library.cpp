#include <random>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>
#include <random>
#include <set>

template <typename iter_type, typename val_type>
bool next_assignment(iter_type start, iter_type end, val_type n) {
    val_type i = *std::prev(end);
    iter_type real_end = std::prev(end);
    iter_type it = std::prev(end);
    iter_type restart_at;
    bool can_restart = false;
    while (it != start) {
        it = std::prev(it);
        if (i - *it > 1) {
            can_restart = true;
            restart_at = it;
            break;
        }
    }
    if (!can_restart) {
        return false;
    }
    *restart_at += 1;
    val_type cur_val = *restart_at;
    while (restart_at != real_end) {
        restart_at = std::next(restart_at);
        *restart_at = cur_val;
    }
    bool valid = true;
    val_type all = 0;
    for (iter_type x = start; x != real_end; x = std::next(x)) {
        all += *x;
        if (*x > *std::next(x)) {
            valid = false;
            break;
        }
    }
    if (!valid) {
        return false;
    }
    *real_end = n - all;
    if (*real_end < *std::prev(real_end)) {
        return false;
    }
    return true;
}

extern "C" int *TravellingSalesmanNaive(const float *distances, int n) {
    auto *result = new int[n];
    std::vector<int> vertices;
    vertices.reserve(n);
    for (int i = 1; i < n; ++i) {
        vertices.push_back(i);
    }
    float min_distance = std::numeric_limits<float>::max();
    do {
        float distance = 0;
        int prev = 0;
        for (int i = 0; i < n - 1; ++i) {
            distance += distances[prev * n + vertices[i]];
            prev = vertices[i];
        }
        distance += distances[prev * n + 0];
        if (distance < min_distance) {
            min_distance = distance;
            result[0] = 0;
            for (int i = 1; i < n; ++i) {
                result[i] = vertices[i - 1];
            }
        }
    } while (std::next_permutation(vertices.begin(), vertices.end()));
    return result;
}

template<typename clazz>
clazz *LoadBalancingTemplate(const clazz *job_exec_times, int n, int worker_num) {
    auto result = new clazz[n + worker_num];
    std::vector<clazz> job_assignment(worker_num, 1);
    job_assignment[worker_num - 1] = n - worker_num + 1;
    auto min_time = std::numeric_limits<clazz>::max();
    std::vector<clazz> job_exec_vec(job_exec_times, job_exec_times + n);
    std::sort(job_exec_vec.begin(), job_exec_vec.end());
    clazz assignment_min_time;
    clazz worker_time;
    do {
        do {
            assignment_min_time = std::numeric_limits<clazz>::min();
            int start = 0;
            for (int i = 0; i < worker_num; ++i) {
                worker_time = 0;
                for (int j = 0; j < job_assignment[i]; ++j) {
                    worker_time += job_exec_vec[start + j];
                }
                if (worker_time > assignment_min_time) {
                    assignment_min_time = worker_time;
                }
                start += job_assignment[i];
            }
            if (assignment_min_time < min_time) {
                min_time = assignment_min_time;
                for (int i = 0; i < worker_num; ++i) {
                    result[i] = job_assignment[i];
                }
                for (int i = worker_num; i < n + worker_num; ++i) {
                    result[i] = job_exec_vec[i - worker_num];
                }
            }
        } while (std::next_permutation(job_exec_vec.begin(), job_exec_vec.end()));
        std::sort(job_exec_vec.begin(), job_exec_vec.end());
    } while (next_assignment(job_assignment.begin(), job_assignment.end(), n));
    return result;
}

extern "C" int *LoadBalancingInt(const int *job_exec_times, int n, int worker_num) {
    return LoadBalancingTemplate(job_exec_times, n, worker_num);
}

extern "C" float *LoadBalancingFloat(const float *job_exec_times, int n, int worker_num) {
    return LoadBalancingTemplate(job_exec_times, n, worker_num);
}

template<typename clazz>
clazz *LoadBalancingGreedyTemplate(const clazz *job_exec_times, int n, int worker_num) {
    // best case + worst case + average load time (float32, when `clazz` is not float32, it will be `reinterpret_cast`ed)
    static_assert(sizeof(clazz) >= sizeof(float), "clazz must be at least as big as float32");
    auto result = new clazz[n + n + 1];
    auto best_time = std::numeric_limits<clazz>::max();
    auto worst_time = std::numeric_limits<clazz>::min();
    std::vector<clazz> job_exec_vec(job_exec_times, job_exec_times + n);
    std::sort(job_exec_vec.begin(), job_exec_vec.end());
    auto arr = new clazz[worker_num];
    clazz zero = 0;
    float avg_time = 0;
    int num_permutations = 0;
    do {
        std::fill(arr, arr + worker_num, zero);
        for (int i = 0; i < n; ++i) {
            int min_worker = 0;
            for (int j = 1; j < worker_num; ++j) {
                if (arr[j] < arr[min_worker]) {
                    min_worker = j;
                }
            }
            arr[min_worker] += job_exec_vec[i];
        }
        clazz max_time = std::numeric_limits<clazz>::min();
        for (int i = 0; i < worker_num; ++i) {
            if (arr[i] > max_time) {
                max_time = arr[i];
            }
        }
        avg_time += max_time;
        ++num_permutations;
        if (max_time < best_time) {
            best_time = max_time;
            for (int i = 0; i < n; ++i) {
                result[i] = job_exec_vec[i];
            }
        }
        if (max_time > worst_time) {
            worst_time = max_time;
            for (int i = 0; i < n; ++i) {
                result[n + i] = job_exec_vec[i];
            }
        }
    } while (std::next_permutation(job_exec_vec.begin(), job_exec_vec.end()));
    delete [] arr;
    avg_time /= float(num_permutations);
    result[n + n] = *reinterpret_cast<clazz*>(&avg_time);
    return result;
}

extern "C" int *LoadBalancingGreedyInt(const int *job_exec_times, int n, int worker_num) {
    return LoadBalancingGreedyTemplate(job_exec_times, n, worker_num);
}

extern "C" float *LoadBalancingGreedyFloat(const float *job_exec_times, int n, int worker_num) {
    return LoadBalancingGreedyTemplate(job_exec_times, n, worker_num);
}

template<typename clazz>
clazz *LoadBalancingDifferentExecTimeTemplate(const clazz *job_exec_times, int n, int worker_num) {
    static_assert(sizeof(clazz) >= sizeof(float), "clazz must be at least as big as float32");
    float order_addon_size_f = float(worker_num * sizeof(int)) / sizeof(clazz);
    int order_addon_size = float(int(order_addon_size_f)) < order_addon_size_f ? int(order_addon_size_f) + 1 : int(order_addon_size_f);
    auto result = new clazz[n + worker_num + order_addon_size];
    std::vector<clazz> job_assignment(worker_num, 0);
    job_assignment[worker_num - 1] = n;
    auto min_time = std::numeric_limits<clazz>::max();
    std::vector<int> job_exec_vec;
    std::vector<int> worker_vec;
    job_exec_vec.reserve(n);
    worker_vec.reserve(worker_num);
    for (int i = 0; i < n; ++i) {
        job_exec_vec.push_back(i);
    }
    for (int i = 0; i < worker_num; ++i) {
        worker_vec.push_back(i);
    }
    std::sort(job_exec_vec.begin(), job_exec_vec.end());
    clazz assignment_min_time;
    clazz worker_time;
    std::vector<int> worker_order;
    do {
        do {
            assignment_min_time = std::numeric_limits<clazz>::min();
            std::sort(worker_vec.begin(), worker_vec.end());
            do {
                int start = 0;
                for (int i = 0; i < worker_num; ++i) {
                    worker_time = 0;
                    for (int j = 0; j < job_assignment[i]; ++j) {
                        auto worker_start = n * worker_vec[i];
                        worker_time += job_exec_times[worker_start + job_exec_vec[start + j]];
                    }
                    if (worker_time > assignment_min_time) {
                        assignment_min_time = worker_time;
                        worker_order = worker_vec;
                    }
                    start += job_assignment[i];
                }
            } while (std::next_permutation(worker_vec.begin(), worker_vec.end()));
            if (assignment_min_time < min_time) {
                min_time = assignment_min_time;
                for (int i = 0; i < worker_num; ++i) {
                    result[i] = job_assignment[i];
                }
                for (int i = worker_num; i < n + worker_num; ++i) {
                    result[i] = job_exec_vec[i - worker_num];
                }
                char *pos = reinterpret_cast<char*>(result + n + worker_num);
                for (int i = 0; i < worker_num; ++i) {
                    int v = worker_order[i];
                    *reinterpret_cast<int*>(pos) = v;
                    pos += sizeof(int);
                }
            }
        } while (std::next_permutation(job_exec_vec.begin(), job_exec_vec.end()));
        std::sort(job_exec_vec.begin(), job_exec_vec.end());
    } while (next_assignment(job_assignment.begin(), job_assignment.end(), n));
    return result;
}

extern "C" int *LoadBalancingDifferentExecTimeInt(const int *job_exec_times, int n, int worker_num) {
    return LoadBalancingDifferentExecTimeTemplate(job_exec_times, n, worker_num);
}

extern "C" float *LoadBalancingDifferentExecTimeFloat(const float *job_exec_times, int n, int worker_num) {
    return LoadBalancingDifferentExecTimeTemplate(job_exec_times, n, worker_num);
}

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

template<typename clazz>
clazz *CenterSelection(const clazz *points, int point_num, int k) {
    auto result = new clazz[k * 2];
    int *idx = new int[k];
    for (int i = 0; i < k; ++i) {
        idx[i] = i;
    }
    float min_dist = std::numeric_limits<float>::max();
    auto min_dist_vec = new float[point_num];
    clazz px, py;
    do {
        std::fill(min_dist_vec, min_dist_vec + point_num, std::numeric_limits<float>::max());
        for (int i = 0; i < k; ++i) {
            min_dist_vec[idx[i]] = 0;
            px = points[idx[i] * 2];
            py = points[idx[i] * 2 + 1];
            float dx, dy, dist;
            for (int j = 0; j < point_num; ++j) {
                if (j == idx[i]) {
                    continue;
                }
                dx = float(px - points[j * 2]);
                dy = float(py - points[j * 2 + 1]);
                dist = dx * dx + dy * dy;
                if (dist < min_dist_vec[j]) {
                    min_dist_vec[j] = dist;
                }
            }
        }
        float max_local_dist = *std::max_element(min_dist_vec, min_dist_vec + point_num);
        if (max_local_dist < min_dist) {
            min_dist = max_local_dist;
            for (int i = 0; i < k; ++i) {
                result[i * 2] = points[idx[i] * 2];
                result[i * 2 + 1] = points[idx[i] * 2 + 1];
            }
        }
    } while (next_combination(idx, point_num, k));
    delete [] idx;
    delete [] min_dist_vec;
    return result;
}

extern "C" int *CenterSelectionInt(const int *points, int point_num, int k) {
    return CenterSelection(points, point_num, k);
}

extern "C" float *CenterSelectionFloat(const float *points, int point_num, int k) {
    return CenterSelection(points, point_num, k);
}

template<typename clazz>
using KMeans2dInitFunc = void (*) (const clazz *, int, float *, int);

template<typename clazz>
using KMedoids2dInitFunc = void (*) (const clazz *, int, int *, int);

template<typename clazz>
using FuzzyCMeansInitFunc = KMeans2dInitFunc<clazz>;

// disable warning "unused-parameter"
template<typename clazz>
void KMedoids2dRandomInit([[maybe_unused]] const clazz *points, int pts_num, int *idx, int k) {
    int *r_idx = new int[pts_num];
    for (int i = 0; i < pts_num; ++i) {
        r_idx[i] = i;
    }
    std::shuffle(r_idx, r_idx + pts_num, std::mt19937(std::random_device()()));
    for (int i = 0; i < k; ++i) {
        idx[i] = r_idx[i];
    }
    delete [] r_idx;
}

template<typename clazz>
void KMeans2dRandomInit(const clazz *points, int pts_num, float *centers, int k) {
    int *idx = new int[k];
    KMedoids2dRandomInit(points, pts_num, idx, k);
    for (int i = 0; i < k; ++i) {
        centers[i * 2] = points[idx[i] * 2];
        centers[i * 2 + 1] = points[idx[i] * 2 + 1];
    }
    delete [] idx;
}

template<typename clazz>
void KMedoids2dKMeansPlusPlus(const clazz *points, int pts_num, int *idx, int k) {
    std::vector<int> idx_vec;
    idx_vec.reserve(k);
    std::mt19937 random_machine{std::random_device()()};
    std::uniform_int_distribution<int> random_int(0, pts_num - 1);
    idx_vec.push_back(random_int(random_machine));
    auto dist_vec = new float[pts_num];
    std::fill(dist_vec, dist_vec + pts_num, std::numeric_limits<float>::max());
    float dx, dy, dist;
    for (int i = 1; i < k; ++i) {
        for (int j = 0; j < pts_num; ++j) {
            if (std::find(idx_vec.begin(), idx_vec.end(), j) != idx_vec.end()) {
                continue;
            }
            dx = float(points[j * 2] - points[idx_vec[i - 1] * 2]);
            dy = float(points[j * 2 + 1] - points[idx_vec[i - 1] * 2 + 1]);
            dist = dx * dx + dy * dy;
            if (dist < dist_vec[j]) {
                dist_vec[j] = dist;
            }
        }
        dist_vec[idx_vec[i - 1]] = 0.0f;
        std::discrete_distribution<int> random_dist(dist_vec, dist_vec + pts_num);
        int new_idx = random_dist(random_machine);
        // avoid duplicate (I don't know if it's necessary,
        // i.e. if the discrete distribution can choose the
        // sample with probability 0)
        if (std::find(idx_vec.begin(), idx_vec.end(), new_idx) != idx_vec.end()) {
            new_idx = random_int(random_machine);
        }
        idx_vec.push_back(new_idx);
    }
    delete [] dist_vec;
    std::copy(idx_vec.begin(), idx_vec.end(), idx);
}

template<typename clazz>
void KMeans2dKMeansPlusPlus(const clazz *points, int pts_num, float *centers, int k) {
    int *idx = new int[k];
    KMedoids2dKMeansPlusPlus(points, pts_num, idx, k);
    for (int i = 0; i < k; ++i) {
        centers[i * 2] = points[idx[i] * 2];
        centers[i * 2 + 1] = points[idx[i] * 2 + 1];
    }
    delete [] idx;
}

template<typename clazz>
float *KMeans2d(const clazz *points, int pts_num, int k, KMeans2dInitFunc<clazz> init_func, float eps, int max_iter) {
    auto centers = new float[k * 2];
    init_func(points, pts_num, centers, k);
    auto new_centers = new float[k * 2];
    auto cluster_size = new int[k];
    auto cluster_sum = new float[k * 2];
    auto dist_vec = new float[k];
    int iter = 0;
    bool converged = false;

    float dx, dy, dist;
    while (!converged && iter < max_iter) {
        std::fill(cluster_size, cluster_size + k, 0);
        std::fill(cluster_sum, cluster_sum + k * 2, 0.0f);
        for (int i = 0; i < pts_num; ++i) {
            for (int j = 0; j < k; ++j) {
                dx = float(points[i * 2] - centers[j * 2]);
                dy = float(points[i * 2 + 1] - centers[j * 2 + 1]);
                dist_vec[j] = dx * dx + dy * dy;
            }
            int min_idx = int(std::min_element(dist_vec, dist_vec + k) - dist_vec);
            cluster_size[min_idx] += 1;
            cluster_sum[min_idx * 2] += points[i * 2];
            cluster_sum[min_idx * 2 + 1] += points[i * 2 + 1];
        }
        for (int i = 0; i < k; ++i) {
            auto cluster_size_f = float(cluster_size[i]);
            new_centers[i * 2] = cluster_sum[i * 2] / cluster_size_f;
            new_centers[i * 2 + 1] = cluster_sum[i * 2 + 1] / cluster_size_f;
        }
        float max_dist = 0.0f;
        for (int i = 0; i < k; ++i) {
            dx = new_centers[i * 2] - centers[i * 2];
            dy = new_centers[i * 2 + 1] - centers[i * 2 + 1];
            dist = dx * dx + dy * dy;
            if (dist > max_dist) {
                max_dist = dist;
            }
        }
        converged = max_dist < eps;
        std::swap(centers, new_centers);
        ++iter;
    }
    delete [] new_centers;
    delete [] cluster_size;
    delete [] cluster_sum;
    delete [] dist_vec;
    auto ret = new float[k * 2 + 2];
    std::copy(centers, centers + k * 2, ret);
    ret[k * 2] = *reinterpret_cast<float*>(&iter);
    ret[k * 2 + 1] = converged ? 1.0f : 0.0f;
    delete [] centers;
    return ret;
}

template<typename clazz>
KMeans2dInitFunc<clazz> KMeansInitMethodSelector(int inti_func) {
    switch (inti_func) {
        case 0:
            return KMeans2dRandomInit<clazz>;
        case 1:
            return KMeans2dKMeansPlusPlus<clazz>;
        default:
            throw std::invalid_argument("Invalid init method");
    }
}

extern "C" float *KMeans2dInt(const int *points, int pts_num, int k, int init_func, float eps, int max_iter) {
    auto init_func_ptr = KMeansInitMethodSelector<int>(init_func);
    return KMeans2d<int>(points, pts_num, k, init_func_ptr, eps, max_iter);
}

extern "C" float *KMeans2dFloat(const float *points, int pts_num, int k, int init_func, float eps, int max_iter) {
    auto init_method_ptr = KMeansInitMethodSelector<float>(init_func);
    return KMeans2d<float>(points, pts_num, k, init_method_ptr, eps, max_iter);
}

template<typename clazz>
int *KMedoids2d(const clazz *points, int pts_num, int k, KMedoids2dInitFunc<clazz> init_func, int max_iter) {
    int *center_idx = new int[k];
    init_func(points, pts_num, center_idx, k);
    auto new_center_idx = new int[k];
    auto label = new int[pts_num];
    int iter = 0;
    bool converged = false;
    auto dist_vec = new float[k];
    while (!converged && iter < max_iter) {
        for (int i = 0; i < pts_num; ++i) {
            for (int j = 0; j < k; ++j) {
                auto dx = float(points[i * 2] - points[center_idx[j] * 2]);
                auto dy = float(points[i * 2 + 1] - points[center_idx[j] * 2 + 1]);
                dist_vec[j] = dx * dx + dy * dy;
            }
            label[i] = int(std::min_element(dist_vec, dist_vec + k) - dist_vec);
        }
        // find new center idx for each cluster
#ifdef OMP_ENABLED
#pragma omp parallel for default(none) shared(points, pts_num, k, new_center_idx, label)
#endif
        for (int i = 0; i < k; ++i) {
            int min_idx = -1;
            float min_dist = std::numeric_limits<float>::max();
            for (int j = 0; j < pts_num; ++j) {
                if (label[j] == i) {
                    float dist = 0.0f;
                    for (int l = 0; l < pts_num; ++l) {
                        if (label[l] == i) {
                            auto dx = float(points[j * 2] - points[l * 2]);
                            auto dy = float(points[j * 2 + 1] - points[l * 2 + 1]);
                            dist += dx * dx + dy * dy;
                        }
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = j;
                    }
                }
            }
            new_center_idx[i] = min_idx;
        }
        converged = std::equal(center_idx, center_idx + k, new_center_idx);
        std::swap(center_idx, new_center_idx);
        ++iter;
    }
    delete [] new_center_idx;
    delete [] dist_vec;
    delete [] label;
    auto ret = new int[k + 2];
    std::copy(center_idx, center_idx + k, ret);
    ret[k] = iter;
    ret[k + 1] = converged ? 1 : 0;
    delete [] center_idx;
    return ret;
}

template<typename clazz>
KMedoids2dInitFunc<clazz> KMedoidsInitMethodSelector(int init_func) {
    switch (init_func) {
        case 0:
            return KMedoids2dRandomInit<clazz>;
        case 1:
            return KMedoids2dKMeansPlusPlus<clazz>;
        default:
            throw std::invalid_argument("Invalid init method");
    }
}

extern "C" int *KMedoids2dInt(const int *points, int pts_num, int k, int init_func, int max_iter) {
    auto init_func_ptr = KMedoidsInitMethodSelector<int>(init_func);
    return KMedoids2d<int>(points, pts_num, k, init_func_ptr, max_iter);
}

extern "C" int *KMedoids2dFloat(const float *points, int pts_num, int k, int init_func, int max_iter) {
    auto init_func_ptr = KMedoidsInitMethodSelector<float>(init_func);
    return KMedoids2d<float>(points, pts_num, k, init_func_ptr, max_iter);
}

template<typename clazz>
float *FuzzyCMeans2d(const clazz *points, int pts_num, int k, float m, FuzzyCMeansInitFunc<clazz> init_func, float eps, int max_iter) {
    auto centers = new float[k * 2];
    init_func(points, pts_num, centers, k);
    auto new_centers = new float[k * 2];
    auto membership = new float[pts_num * k];
    auto new_membership = new float[pts_num * k];
    auto dist_square_mat = new float[pts_num * k];
    // init membership, set the membership of the closest center to 1.0
    for (int i = 0; i < pts_num; ++i) {
        float min_dist = std::numeric_limits<float>::max();
        int min_idx = -1;
        for (int j = 0; j < k; ++j) {
            auto dx = float(points[i * 2] - centers[j * 2]);
            auto dy = float(points[i * 2 + 1] - centers[j * 2 + 1]);
            auto dist_square = dx * dx + dy * dy;
            dist_square_mat[i * k + j] = dist_square;
            if (dist_square < min_dist) {
                min_dist = dist_square;
                min_idx = j;
            }
        }
        for (int j = 0; j < k; ++j) {
            membership[i * k + j] = 0.0f;
        }
        membership[i * k + min_idx] = 1.0f;
    }

    int iter = 0;
    bool converged = false;
    while (!converged && iter < max_iter) {
        // update centers
#ifdef OMP_ENABLED
#pragma omp parallel for default(none) shared(new_centers, membership, points, k, m, pts_num)
#endif
        for (int i = 0; i < k; ++i) {
            float sum_num_x = 0.0f;
            float sum_num_y = 0.0f;
            float sum_den = 0.0f;
            for (int j = 0; j < pts_num; ++j) {
                auto mem = std::pow(membership[j * k + i], m);
                sum_num_x += mem * points[j * 2];
                sum_num_y += mem * points[j * 2 + 1];
                sum_den += mem;
            }
            new_centers[i * 2] = sum_num_x / sum_den;
            new_centers[i * 2 + 1] = sum_num_y / sum_den;
        }
        // update membership
#ifdef OMP_ENABLED
#pragma omp parallel for default(none) shared(dist_square_mat, points, new_centers, k, pts_num)
#endif
        for (int i = 0; i < pts_num; ++i) {
            for (int j = 0; j < k; ++j) {
                auto dx = float(points[i * 2] - new_centers[j * 2]);
                auto dy = float(points[i * 2 + 1] - new_centers[j * 2 + 1]);
                dist_square_mat[i * k + j] = dx * dx + dy * dy;
            }
        }
#ifdef OMP_ENABLED
#pragma omp parallel for default(none) shared(new_membership, dist_square_mat, k, m, pts_num)
#endif
        for (int i = 0; i < pts_num; ++i) {
            for (int j = 0; j < k; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    sum += std::pow(dist_square_mat[i * k + j] / dist_square_mat[i * k + l], 1.0f / (m - 1.0f));
                }
                new_membership[i * k + j] = 1.0f / sum;
            }
        }
        // norm membership for numerical stability
        for (int i = 0; i < pts_num; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < k; ++j) {
                sum += new_membership[i * k + j];
            }
            for (int j = 0; j < k; ++j) {
                new_membership[i * k + j] /= sum;
            }
        }
        // check convergence
        float max_dist = 0.0f;
        for (int i = 0; i < k; ++i) {
            auto dx = new_centers[i * 2] - centers[i * 2];
            auto dy = new_centers[i * 2 + 1] - centers[i * 2 + 1];
            auto dist = dx * dx + dy * dy;
            if (dist > max_dist) {
                max_dist = dist;
            }
        }
        for (int i = 0; i < pts_num * k; ++i) {
            auto dist = std::abs(membership[i] - new_membership[i]);
            if (dist > max_dist) {
                max_dist = dist;
            }
        }
        converged = max_dist < eps;
        std::swap(centers, new_centers);
        std::swap(membership, new_membership);
        ++iter;
    }
    delete [] new_centers;
    delete [] new_membership;
    delete [] dist_square_mat;
    auto ret = new float[k * (2 + pts_num) + 2];
    std::copy(centers, centers + k * 2, ret);
    std::copy(membership, membership + pts_num * k, ret + k * 2);
    ret[k * (2 + pts_num)] = *reinterpret_cast<float *>(&iter);
    ret[k * (2 + pts_num) + 1] = converged ? 1.0f : 0.0f;
    delete [] centers;
    delete [] membership;
    return ret;
}

template<typename clazz>
FuzzyCMeansInitFunc<clazz> FuzzyCMeansInitMethodSelector(int init_func) {
    return KMeansInitMethodSelector<clazz>(init_func);
}

extern "C" float *FuzzyCMeans2dInt(const int *points, int pts_num, int k, float m, int init_func, float eps, int max_iter) {
    auto init_func_ptr = FuzzyCMeansInitMethodSelector<int>(init_func);
    return FuzzyCMeans2d<int>(points, pts_num, k, m, init_func_ptr, eps, max_iter);
}

extern "C" float *FuzzyCMeans2dFloat(const float *points, int pts_num, int k, float m, int init_func, float eps, int max_iter) {
    auto init_func_ptr = FuzzyCMeansInitMethodSelector<float>(init_func);
    return FuzzyCMeans2d<float>(points, pts_num, k, m, init_func_ptr, eps, max_iter);
}

template<typename clazz>
clazz *SetCover(const int *subsets, const int *subset_lens, const clazz *subset_weights, int *order, int subset_len_len) {
    static_assert(sizeof(clazz) >= sizeof(int), "clazz must be at least as big as int32");
    auto ret = new clazz[2];
    std::set<int> all_elements;
    int total_len = std::accumulate(subset_lens, subset_lens + subset_len_len, 0);
    for (int i = 0; i < total_len; ++i) {
        all_elements.insert(subsets[i]);
    }
    int num_elements = int(all_elements.size());
    char *covered = new char[num_elements];
    std::fill(covered, covered + num_elements, char(0));
    std::vector<int> cur_order;
    std::vector<std::vector<int>> ordered_subsets;
    ordered_subsets.reserve(subset_len_len);
    int *start = const_cast<int *>(subsets);
    for (int i = 0; i < subset_len_len; ++i) {
        ordered_subsets.emplace_back(start, start + subset_lens[i]);
        start += subset_lens[i];
    }
    cur_order.reserve(subset_len_len);
    for (int i = 0; i < subset_len_len; ++i) {
        cur_order.push_back(i);
    }
    clazz min_cost = std::numeric_limits<clazz>::max();
    do {
        std::fill(covered, covered + num_elements, char(0));
        clazz cur_weight = 0.0f;
        int end = subset_len_len;
        for (int i = 0; i < subset_len_len; ++i) {
            if (std::accumulate(covered, covered + num_elements, 0) == num_elements) {
                end = i;
                break;
            }
            auto &subset = ordered_subsets[cur_order[i]];
            for (const auto & element : subset) {
                if (covered[element] == 0) {
                    covered[element] = 1;
                }
            }
            cur_weight += subset_weights[cur_order[i]];
        }
        std::reverse(cur_order.begin() + end, cur_order.end());
        if (cur_weight < min_cost) {
            min_cost = cur_weight;
            ret[0] = min_cost;
            ret[1] = *reinterpret_cast<clazz *>(&end);
            std::copy(cur_order.begin(), cur_order.end(), order);
        }
    } while (std::next_permutation(cur_order.begin(), cur_order.end()));
    delete [] covered;
    return ret;
}

extern "C" int *SetCoverInt(const int *subsets, const int *subset_lens, const int *subset_weights, int *order, int subset_len_len) {
    return SetCover<int>(subsets, subset_lens, subset_weights, order, subset_len_len);
}

extern "C" float *SetCoverFloat(const int *subsets, const int *subset_lens, const float *subset_weights, int *order, int subset_len_len) {
    return SetCover<float>(subsets, subset_lens, subset_weights, order, subset_len_len);
}

int PricingInnerInt(int const * const * const edge, int len_edge, const int *weight, int len_weight, int *arr, int *nodes, bool *node_done, bool *edge_done) {
    std::copy(weight, weight + len_weight, nodes);
    std::fill(node_done, node_done + len_weight, false);
    std::fill(edge_done, edge_done + len_edge, false);
    while (!std::all_of(edge_done, edge_done + len_edge, [](bool b) { return b; })) {
        for (int i = 0; i < len_edge; ++i) {
            if (edge_done[i]) {
                continue;
            }
            edge_done[i] = true;
            if (node_done[edge[i][0]] or node_done[edge[i][1]]) {
                continue;
            }
            if (nodes[edge[i][0]] == nodes[edge[i][1]]) {
                node_done[edge[i][0]] = true;
                node_done[edge[i][1]] = true;
                continue;
            }
            if (nodes[edge[i][0]] < nodes[edge[i][1]]) {
                nodes[edge[i][1]] -= nodes[edge[i][0]];
                node_done[edge[i][0]] = true;
            } else {
                nodes[edge[i][0]] -= nodes[edge[i][1]];
                node_done[edge[i][1]] = true;
            }
        }
    }
    int ret = 0;
    for (int i = 0; i < len_weight; ++i) {
        if (node_done[i]) {
            ret += weight[i];
        }
    }
    std::copy(node_done, node_done + len_weight, arr);
    return ret;
}

extern "C" int *PricingMethodInt(const int *edge, int len_edge, const int *weights, int len_weight, int *best, int *worst) {
    int *ret = new int[2];
    int best_cost = std::numeric_limits<int>::max();
    int worst_cost = std::numeric_limits<int>::min();
    int *node = new int[len_weight];
    bool *node_done = new bool[len_weight];
    bool *edge_done = new bool[len_edge];
    int *arr = new int[len_weight];
    int ** const edge_wrapper = new int *[len_edge];
    for (int i = 0; i < len_edge; ++i) {
        edge_wrapper[i] = const_cast<int *>(edge + i * 2);
    }
    std::sort(edge_wrapper, edge_wrapper + len_edge);
    do {
        int cost = PricingInnerInt(edge_wrapper, len_edge, weights, len_weight, arr, node, node_done, edge_done);
        if (cost < best_cost) {
            best_cost = cost;
            std::copy(arr, arr + len_weight, best);
        }
        if (cost > worst_cost) {
            worst_cost = cost;
            std::copy(arr, arr + len_weight, worst);
        }
    } while (std::next_permutation(edge_wrapper, edge_wrapper + len_edge));
    ret[0] = best_cost;
    ret[1] = worst_cost;
    delete [] node;
    delete [] node_done;
    delete [] edge_done;
    delete [] arr;
    delete [] edge_wrapper;
    return ret;
}

float PricingInnerFloat(int const * const * const edge, int len_edge, const float *weight, int len_weight, int *arr, int *nodes, bool *node_done, bool *edge_done) {
    std::copy(weight, weight + len_weight, nodes);
    std::fill(node_done, node_done + len_weight, false);
    std::fill(edge_done, edge_done + len_edge, false);
    while (!std::all_of(edge_done, edge_done + len_edge, [](bool b) { return b; })) {
        for (int i = 0; i < len_edge; ++i) {
            if (edge_done[i]) {
                continue;
            }
            edge_done[i] = true;
            if (node_done[edge[i][0]] or node_done[edge[i][1]]) {
                continue;
            }
            if (std::abs(nodes[edge[i][0]] - nodes[edge[i][1]]) < 1e-6) {
                node_done[edge[i][0]] = true;
                node_done[edge[i][1]] = true;
                continue;
            }
            if (nodes[edge[i][0]] < nodes[edge[i][1]]) {
                nodes[edge[i][1]] -= nodes[edge[i][0]];
                node_done[edge[i][0]] = true;
            } else {
                nodes[edge[i][0]] -= nodes[edge[i][1]];
                node_done[edge[i][1]] = true;
            }
        }
    }
    float ret = 0;
    for (int i = 0; i < len_weight; ++i) {
        if (node_done[i]) {
            ret += weight[i];
        }
    }
    std::copy(node_done, node_done + len_weight, arr);
    return ret;
}

extern "C" float *PricingMethodFloat(const int *edge, int len_edge, const float *weights, int len_weight, int *best, int *worst) {
    float *ret = new float[2];
    float best_cost = std::numeric_limits<float>::max();
    float worst_cost = std::numeric_limits<float>::min();
    int *node = new int[len_weight];
    bool *node_done = new bool[len_weight];
    bool *edge_done = new bool[len_edge];
    int *arr = new int[len_weight];
    int ** const edge_wrapper = new int *[len_edge];
    for (int i = 0; i < len_edge; ++i) {
        edge_wrapper[i] = const_cast<int *>(edge + i * 2);
    }
    std::sort(edge_wrapper, edge_wrapper + len_edge);
    do {
        float cost = PricingInnerFloat(edge_wrapper, len_edge, weights, len_weight, arr, node, node_done, edge_done);
        if (cost < best_cost) {
            best_cost = cost;
            std::copy(arr, arr + len_weight, best);
        }
        if (cost > worst_cost) {
            worst_cost = cost;
            std::copy(arr, arr + len_weight, worst);
        }
    } while (std::next_permutation(edge_wrapper, edge_wrapper + len_edge));
    ret[0] = best_cost;
    ret[1] = worst_cost;
    delete [] node;
    delete [] node_done;
    delete [] edge_done;
    delete [] arr;
    delete [] edge_wrapper;
    return ret;
}

template <typename T>
bool IsInVector(const std::vector<T> &vec, const T &val) {
    return std::find(vec.begin(), vec.end(), val) != vec.end();
}

struct DisjointPathGraph {
    int vertex_num = 0;
    int edge_capacity_value = 1;
    std::vector<std::pair<int, int>> edges;
    std::vector<int> source_vertex;
    std::vector<int> target_vertex;
    std::vector<std::pair<int, std::vector<int>>> adj_list;
    std::vector<int> edge_capacity;

    // internal variables
    std::vector<int> _dist;
    std::vector<int> _prev;
    std::vector<char> _visited;
    DisjointPathGraph() = default;
    DisjointPathGraph(int vertex_num, int edge_capacity_value,
                      const int *edges, int edge_num,
                      const int *source_vertex,
                      const int *target_vertex,
                      int source_target_len):
    vertex_num(vertex_num), edge_capacity_value(edge_capacity_value) {
        this->edges.reserve(edge_num);
        for (int i = 0; i < edge_num; ++i) {
            this->edges.emplace_back(edges[i * 2], edges[i * 2 + 1]);
        }
        this->source_vertex = std::vector<int>(source_vertex, source_vertex + source_target_len);
        this->target_vertex = std::vector<int>(target_vertex, target_vertex + source_target_len);
        this->edge_capacity = std::vector<int>(edge_num, edge_capacity_value);
        buildAdjList();
    }

    void buildAdjList() {
        adj_list.resize(vertex_num);
        for (int i = 0; i < vertex_num; ++i) {
            adj_list[i].first = i;
            adj_list[i].second.clear();
        }
        for (const auto &edge: edges) {
            adj_list[edge.first].second.push_back(edge.second);
        }
        std::fill(edge_capacity.begin(), edge_capacity.end(), edge_capacity_value);
        _dist.resize(vertex_num);
        _prev.resize(vertex_num);
        _visited.resize(vertex_num);
    }

    [[nodiscard]] bool isValidVertex(int vertex) const {
        if (vertex < 0 or vertex >= vertex_num) {
            return false;
        }
        return true;
    }

    bool addEdge(int source, int target, bool rebuild = true) {
        if (!isValidVertex(source) or !isValidVertex(target)) {
            return false;
        }
        edges.emplace_back(source, target);
        if (rebuild) {
            buildAdjList();
        }
        return true;
    }

    bool addSourceTarget(int source, int target, bool rebuild = true) {
        // make sure source and target not in current sources and targets
        if (!isValidVertex(source) or !isValidVertex(target)) {
            return false;
        }
        if (IsInVector(source_vertex, source) or IsInVector(target_vertex, source)) {
            return false;
        }
        if (IsInVector(source_vertex, target) or IsInVector(target_vertex, target)) {
            return false;
        }
        source_vertex.push_back(source);
        target_vertex.push_back(target);
        if (rebuild) {
            buildAdjList();
        }
        return true;
    }

    void disableEdgeByOne(int in, int out) {
        auto it = std::find(edges.begin(), edges.end(), std::make_pair(in, out));
        if (it != edges.end()) {
            auto idx = std::distance(edges.begin(), it);
            if (edge_capacity[idx] > 0) {
                edge_capacity[idx] -= 1;
            }
            if (edge_capacity[idx] == 0) {
                adj_list[in].second.erase(std::find(adj_list[in].second.begin(), adj_list[in].second.end(), out));
            }
        }
    }

    void resetEdgeCapacity() {
        std::fill(edge_capacity.begin(), edge_capacity.end(), edge_capacity_value);
        for (int i = 0; i < vertex_num; ++i) {
            adj_list[i].second.clear();
        }
        for (const auto &edge: edges) {
            adj_list[edge.first].second.push_back(edge.second);
        }
    }

    [[nodiscard]] std::vector<std::pair<int, int>> findShortestPath(int which) {
        int source = source_vertex[which];
        int target = target_vertex[which];
        std::vector<std::pair<int, int>> ret;
        // Dijkstra with directed graph
        std::fill(_dist.begin(), _dist.end(), std::numeric_limits<int>::max());
        std::fill(_prev.begin(), _prev.end(), -1);
        std::fill(_visited.begin(), _visited.end(), 0);
        _dist[source] = 0;
        for (int i = 0; i < vertex_num; ++i) {
            int min_dist = std::numeric_limits<int>::max();
            int min_dist_vertex = -1;
            for (int j = 0; j < vertex_num; ++j) {
                if (!_visited[j] and _dist[j] < min_dist) {
                    min_dist = _dist[j];
                    min_dist_vertex = j;
                }
            }
            if (min_dist_vertex == -1 or min_dist_vertex == target) {
                break;
            }
            _visited[min_dist_vertex] = 1;
            for (size_t j = 0; j < adj_list[min_dist_vertex].second.size(); ++j) {
                int next_vertex = adj_list[min_dist_vertex].second[j];
                if (_dist[next_vertex] > _dist[min_dist_vertex] + 1) {
                    _dist[next_vertex] = _dist[min_dist_vertex] + 1;
                    _prev[next_vertex] = min_dist_vertex;
                }
            }
        }
        if (_dist[target] == std::numeric_limits<int>::max()) {
            return ret;
        }
        int cur = target;
        while (cur != source) {
            ret.emplace_back(_prev[cur], cur);
            cur = _prev[cur];
        }
        std::reverse(ret.begin(), ret.end());
        return ret;
    }

};

extern "C" int* DisjointPathProblemC1(
        int vertex_num,
        const int *edges, int edge_num,
        const int *source_vertex,
        const int *target_vertex,
        int source_target_len) {

    DisjointPathGraph graph(vertex_num, 1, edges, edge_num, source_vertex, target_vertex, source_target_len);
    std::vector<std::vector<std::pair<int, int>>> best_paths;
    std::vector<std::vector<std::pair<int, int>>> worst_paths;
    std::vector<std::vector<std::pair<int, int>>> cur_paths;
    std::vector<int> order(source_target_len);
    std::iota(order.begin(), order.end(), 0);
    std::vector<char> visited(source_target_len, 0);
    std::vector<int> best_order;
    std::vector<int> worst_order;
    do {
        std::fill(visited.begin(), visited.end(), 0);
        cur_paths.clear();
        graph.resetEdgeCapacity();
        for (int i = 0; i < source_target_len; ++i) {
            auto cur_path = graph.findShortestPath(order[i]);
            if (cur_path.empty()) {
                continue;
            }
            for (const auto &edge: cur_path) {
                graph.disableEdgeByOne(edge.first, edge.second);
            }
            visited[i] = 1;
            cur_paths.push_back(cur_path);
        }
        // re arrange order
//        std::vector<int> new_order;
//        for (int i = 0; i < source_target_len; ++i) {
//            if (visited[i]) {
//                new_order.push_back(order[i]);
//            }
//        }
//        for (int i = 0; i < source_target_len; ++i) {
//            if (!visited[i]) {
//                new_order.push_back(order[i]);
//            }
//        }
//        order = new_order;
        if (cur_paths.size() > best_paths.size()) {
            best_paths = cur_paths;
            best_order = order;
        }
        if (cur_paths.size() < worst_paths.size() or worst_paths.empty()) {
            worst_paths = cur_paths;
            worst_order = order;
        }
    } while (std::next_permutation(order.begin(), order.end()));
    int best_path_len = 0;
    int worst_path_len = 0;
    for (const auto &path: best_paths) {
        best_path_len += path.size();
    }
    for (const auto &path: worst_paths) {
        worst_path_len += path.size();
    }
    int ret_size = 2 + 2 * (source_target_len + 2 * best_path_len + 2 * worst_path_len);
    int *ret = new int[ret_size];
    ret[0] = best_path_len;
    ret[1] = worst_path_len;
    std::copy(best_order.begin(), best_order.end(), ret + 2);
    std::copy(worst_order.begin(), worst_order.end(), ret + 2 + source_target_len);
    int cur = 2 + 2 * source_target_len;
    for (const auto &path: best_paths) {
        for (const auto &edge: path) {
            ret[cur++] = edge.first;
            ret[cur++] = edge.second;
        }
    }
    for (const auto &path: worst_paths) {
        for (const auto &edge: path) {
            ret[cur++] = edge.first;
            ret[cur++] = edge.second;
        }
    }
    return ret;
}
