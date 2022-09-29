#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>

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

//extern "C" int *LoadBalancingInt(const int *job_exec_times, int n, int worker_num) {
//    int *result = new int[n + worker_num];
//    std::vector<int> job_assignment(worker_num, 1);
//    job_assignment[worker_num - 1] = n - worker_num + 1;
//    int min_time = std::numeric_limits<int>::max();
//    std::vector<int> job_exec_vec(job_exec_times, job_exec_times + n);
//    std::sort(job_exec_vec.begin(), job_exec_vec.end());
//    int assignment_min_time;
//    int worker_time;
//    do {
//        do {
//            assignment_min_time = std::numeric_limits<int>::min();
//            int start = 0;
//            for (int i = 0; i < worker_num; ++i) {
//                worker_time = 0;
//                for (int j = 0; j < job_assignment[i]; ++j) {
//                    worker_time += job_exec_vec[start + j];
//                }
//                if (worker_time > assignment_min_time) {
//                    assignment_min_time = worker_time;
//                }
//                start += job_assignment[i];
//            }
//            if (assignment_min_time < min_time) {
//                min_time = assignment_min_time;
//                for (int i = 0; i < worker_num; ++i) {
//                    result[i] = job_assignment[i];
//                }
//                for (int i = worker_num; i < n + worker_num; ++i) {
//                    result[i] = job_exec_vec[i - worker_num];
//                }
//            }
//        } while (std::next_permutation(job_exec_vec.begin(), job_exec_vec.end()));
//        std::sort(job_exec_vec.begin(), job_exec_vec.end());
//    } while (next_assignment(job_assignment.begin(), job_assignment.end(), n));
//    return result;
//}

extern "C" int *LoadBalancingInt(const int *job_exec_times, int n, int worker_num) {
    return LoadBalancingTemplate(job_exec_times, n, worker_num);
}

extern "C" float *LoadBalancingFloat(const float *job_exec_times, int n, int worker_num) {
    return LoadBalancingTemplate(job_exec_times, n, worker_num);
}

template<typename clazz>
clazz *LoadBalancingGreedyTemplate(const clazz *job_exec_times, int n, int worker_num) {
    // best case + worst case
    auto result = new clazz[n + n];
    auto best_time = std::numeric_limits<clazz>::max();
    auto worst_time = std::numeric_limits<clazz>::min();
    std::vector<clazz> job_exec_vec(job_exec_times, job_exec_times + n);
    std::sort(job_exec_vec.begin(), job_exec_vec.end());
    auto arr = new clazz[worker_num];
    clazz zero = 0;
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
    return result;
}


//extern "C" int *LoadBalancingGreedyInt(const int *job_exec_times, int n, int worker_num) {
//    // best case + worst case
//    int *result = new int[n + n];
//    int best_time = std::numeric_limits<int>::max();
//    int worst_time = std::numeric_limits<int>::min();
//    std::vector<int> job_exec_vec(job_exec_times, job_exec_times + n);
//    std::sort(job_exec_vec.begin(), job_exec_vec.end());
//    int *arr = new int[worker_num];
//    do {
//        std::fill(arr, arr + worker_num, 0);
//        for (int i = 0; i < n; ++i) {
//            int min_worker = 0;
//            for (int j = 1; j < worker_num; ++j) {
//                if (arr[j] < arr[min_worker]) {
//                    min_worker = j;
//                }
//            }
//            arr[min_worker] += job_exec_vec[i];
//        }
//        int max_time = std::numeric_limits<int>::min();
//        for (int i = 0; i < worker_num; ++i) {
//            if (arr[i] > max_time) {
//                max_time = arr[i];
//            }
//        }
//        if (max_time < best_time) {
//            best_time = max_time;
//            for (int i = 0; i < n; ++i) {
//                result[i] = job_exec_vec[i];
//            }
//        }
//        if (max_time > worst_time) {
//            worst_time = max_time;
//            for (int i = 0; i < n; ++i) {
//                result[n + i] = job_exec_vec[i];
//            }
//        }
//    } while (std::next_permutation(job_exec_vec.begin(), job_exec_vec.end()));
//    delete [] arr;
//    return result;
//}

extern "C" int *LoadBalancingGreedyInt(const int *job_exec_times, int n, int worker_num) {
    return LoadBalancingGreedyTemplate(job_exec_times, n, worker_num);
}

extern "C" float *LoadBalancingGreedyFloat(const float *job_exec_times, int n, int worker_num) {
    return LoadBalancingGreedyTemplate(job_exec_times, n, worker_num);
}
