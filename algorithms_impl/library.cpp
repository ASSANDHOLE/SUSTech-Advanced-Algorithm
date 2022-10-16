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
    delete[] idx;
    delete[] min_dist_vec;
    return result;
}

extern "C" int *CenterSelectionInt(const int *points, int point_num, int k) {
    return CenterSelection(points, point_num, k);
}

extern "C" float *CenterSelectionFloat(const float *points, int point_num, int k) {
    return CenterSelection(points, point_num, k);
}
