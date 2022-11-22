#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <fstream>
#include <vector>
#include <utility>
#include <tuple>

#include <ortools/linear_solver/linear_solver.h>

std::string kPythonConvertPath;
bool do_print = true;

#define NO_PRINT do_print = false;
#define DO_PRINT do_print = true;

#define TIME_INIT auto _time_start = std::chrono::steady_clock::now(); \
auto _time_end = _time_start; \
auto _duration = 0L;

#define TIME_START _time_start = std::chrono::steady_clock::now(); NO_PRINT;

#define TIME_END(NAME) _time_end = std::chrono::steady_clock::now(); \
_duration = std::chrono::duration_cast<std::chrono::milliseconds>(_time_end - _time_start).count(); \
std::cout << (NAME) << " -- Duration: " << _duration << "ms" << std::endl; DO_PRINT;

namespace data_utils {

    struct Data {
        Data(std::string type_name, int d0, int d1):
                d0(d0), d1(d1) {
            int type_size = 0;
            if (type_name == "ui8" || type_name == "i8") {
                type_size = 1;
            } else if (type_name == "ui16" || type_name == "i16") {
                type_size = 2;
            } else if (type_name == "ui32" || type_name == "i32" || type_name == "f32") {
                type_size = 4;
            } else if (type_name == "ui64" || type_name == "i64" || type_name == "f64") {
                type_size = 8;
            }
            this->type_name = std::move(type_name);
            this->type_size = type_size;
            this->arr = new char[d0 * d1 * type_size];
        }

        Data(Data &&other) noexcept {
            this->type_name = std::move(other.type_name);
            this->type_size = other.type_size;
            this->d0 = other.d0;
            this->d1 = other.d1;
            this->arr = other.arr;
            other.arr = nullptr;
        }

        ~Data() {
            delete [] arr;
        }

        template<typename T>
        [[nodiscard]] T get(int x, int y) const {
            int index = x * d1 + y;
            if (type_name == "ui8") {
                return T(((unsigned char *) arr)[index]);
            } else if (type_name == "i8") {
                return T(((char *) arr)[index]);
            } else if (type_name == "ui16") {
                return T(((unsigned short *) arr)[index]);
            } else if (type_name == "i16") {
                return T(((short *) arr)[index]);
            } else if (type_name == "ui32") {
                return T(((unsigned int *) arr)[index]);
            } else if (type_name == "i32") {
                return T(((int *) arr)[index]);
            } else if (type_name == "ui64") {
                return T(((unsigned long *) arr)[index]);
            } else if (type_name == "i64") {
                return T(((long *) arr)[index]);
            } else if (type_name == "f32") {
                return T(((float *) arr)[index]);
            } else if (type_name == "f64") {
                return T(((double *) arr)[index]);
            }
            return 0;
        }

        template<typename T>
        void set(int x, int y, T value) {
            int index = x * d1 + y;
            if (type_name == "ui8") {
                ((unsigned char *) arr)[index] = value;
            } else if (type_name == "i8") {
                ((char *) arr)[index] = value;
            } else if (type_name == "ui16") {
                ((unsigned short *) arr)[index] = value;
            } else if (type_name == "i16") {
                ((short *) arr)[index] = value;
            } else if (type_name == "ui32") {
                ((unsigned int *) arr)[index] = value;
            } else if (type_name == "i32") {
                ((int *) arr)[index] = value;
            } else if (type_name == "ui64") {
                ((unsigned long *) arr)[index] = value;
            } else if (type_name == "i64") {
                ((long *) arr)[index] = value;
            } else if (type_name == "f32") {
                ((float *) arr)[index] = value;
            } else if (type_name == "f64") {
                ((double *) arr)[index] = value;
            }
        }

        std::string type_name;
        int type_size;
        int d1;
        int d0;
        char *arr;
    };

    void ConvertMatlabMatToBinary(const std::string &python_executable,
                                  const std::string &matlab_mat_file,
                                  const std::string &binary_dir) {
        std::string command = python_executable + " " + kPythonConvertPath + " " + matlab_mat_file + " " + binary_dir;
        std::cout << "Running command: " << command << std::endl;
        int status = system(command.c_str());
        if (status != 0) {
            std::cout << "Error running command: " << command << "\nExit status: " << status << std::endl;
            exit(1);
        }
    }

    auto LoadData(const std::string &binary_dir) {
        // contents: {NAME}:({shape},):{data_type}
#ifdef _WIN32
        const char delim = '\\';
#else
        const char delim = '/';
#endif
        std::ifstream meta_info(binary_dir + "/keys.txt");
        std::string line;
        std::vector<std::tuple<std::string, Data>> data;
        while (std::getline(meta_info, line)) {
            std::string name;
            std::string shape;
            std::string data_type;
            int i = 0;
            for (; i < line.size(); ++i) {
                if (line[i] == ':') {
                    break;
                }
                name += line[i];
            }
            ++i;
            for (; i < line.size(); ++i) {
                if (line[i] == ':') {
                    break;
                }
                shape += line[i];
            }
            ++i;
            for (; i < line.size(); ++i) {
                if (line[i] == ':') {
                    break;
                }
                data_type += line[i];
            }
            int d0 = std::stoi(shape.substr(1, shape.find(' ') - 1));
            int d1 = std::stoi(shape.substr(shape.find(' ') + 1, shape.size() - shape.find(' ') - 2));
            std::cout << "Loading " << name << " with shape (" << d0 << ", " << d1 << ") and data type " << data_type << std::endl;
            std::string binary_dir_with_delim = binary_dir;
            if (binary_dir_with_delim.back() != delim) {
                binary_dir_with_delim += delim;
            }
            std::ifstream data_file(binary_dir_with_delim + name + ".bin", std::ios::binary);
            Data data_obj(data_type, d0, d1);
            auto read_len = d0 * d1 * data_obj.type_size;
            auto *data_arr = new char[read_len];
            data_file.read(data_arr, read_len);
            memcpy(data_obj.arr, data_arr, read_len);
            data.emplace_back(name, std::move(data_obj));
        }
        return data;
    }

    void RemoveDir(const std::string &dir) {
#ifdef _WIN32
        std::string command = "rmdir /s /q " + dir;
#else
        std::string command = "rm -rf " + dir;
#endif
        std::cout << "Running command: " << command << std::endl;
        int status = system(command.c_str());
        if (status != 0) {
            std::cout << "Error running command: " << command << "\nExit status: " << status << std::endl;
            exit(1);
        }
    }
} // namespace data_util

namespace operations_research {

    void Print(auto &&data, bool new_line = true) {
        if (!do_print) {
            return;
        }
        std::cout << data;
        if (new_line) {
            std::cout << std::endl;
        }
    }

    void LinearProgramming(const std::string &python_executable,
                           const std::string &matlab_mat_file,
                           const std::string &binary_dir,
                           const std::string &solver_name) {
        std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver(solver_name));
        if (!solver) {
            Print(solver_name + " solver unavailable.");
            return;
        } else {
            Print("Using " + solver_name + " solver.");
        }
        solver->SetNumThreads(10);

        data_utils::ConvertMatlabMatToBinary(python_executable, matlab_mat_file, binary_dir);
        const auto data = data_utils::LoadData(binary_dir);
        const double infinity = solver->infinity();
        // x and y are non-negative variables.
        std::vector<MPVariable*> x;
        const data_utils::Data *mat_a;
        const data_utils::Data *mat_b;
        const data_utils::Data *mat_c;
        // matrix A: (1, n)
        // matrix B: (m, n)
        // matrix C: (m, 1)

        for (const auto & i : data) {
            if (std::get<0>(i) == "A") {
                mat_a = &std::get<1>(i);
            }
            if (std::get<0>(i) == "B") {
                mat_b = &std::get<1>(i);
            }
            if (std::get<0>(i) == "C") {
                mat_c = &std::get<1>(i);
            }
        }

        TIME_INIT
        TIME_START
        // init x
        for (int j = 0; j < mat_a->d1; ++j) {
            x.push_back(solver->MakeNumVar(0.0, infinity, "x"+std::to_string(j)));
        }

        Print("Number of variables = ", false); Print(solver->NumVariables());

        // init constraints with b[i] .* x <= c[i]
        for (int i = 0; i < mat_b->d0; ++i) {
            MPConstraint *const ct = solver->MakeRowConstraint(-infinity, mat_c->get<int32_t>(1, 0));
            for (int j = 0; j < mat_b->d1; ++j) {
                ct->SetCoefficient(x[j], double_t(mat_b->get<int64_t>(i, j)));
            }
        }

        // init objective function with A[0] .* x
        MPObjective *const objective = solver->MutableObjective();
        for (int j = 0; j < mat_a->d1; ++j) {
            objective->SetCoefficient(x[j], double_t(mat_a->get<int64_t>(0, j)));
        }

        objective->SetMaximization();

        TIME_END("Init Problem")

        TIME_START
        const MPSolver::ResultStatus result_status = solver->Solve();
        TIME_END("Solve Problem")

        // Check that the problem has an optimal solution.
        if (result_status != MPSolver::OPTIMAL) {
            Print("The problem does not have an optimal solution!");
            return;
        }

        Print("Solution:");
        Print("Optimal objective value = ", false); Print(objective->Value());
    }
}  // namespace operations_research

int main(int argc, char** argv) {
#define use_flags

#ifdef use_flags
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <python_executable> [tmp_dir] [<problem_idx> <solver_idx>]" << std::endl;
        return 1;
    }
    auto python_executable = argv[1];
#else
    auto python_executable = "/usr/bin/python3";
#endif
    std::string tmp_dir;
    if (argc > 2) {
        tmp_dir = argv[2];
    } else {
        tmp_dir = "./tmp";
    }
    auto current_dir = std::string(__FILE__);

#ifdef _WIN32
    current_dir = current_dir.substr(0, current_dir.find_last_of('\\'));
#else
    current_dir = current_dir.substr(0, current_dir.find_last_of('/'));
#endif
    using std::literals::string_literals::operator""s;
    std::string problems []{"small"s, "medium"s, "large"s};
    std::string solvers []{"CBC"s, "SCIP"s, "GLOP"s, "GUROBI"s, "CPLEX"s};

    int problem_index = 0;
    int solver_index = 0;

#ifdef use_flags
    problem_index = std::stoi(argv[3]);
    solver_index = std::stoi(argv[4]);
    printf("problem_index: %d, solver_index: %d\n", problem_index, solver_index);
#endif

    kPythonConvertPath = std::string(current_dir + "/convert_matlab_mat_file.py");
    operations_research::LinearProgramming(
        python_executable,
        std::string(current_dir + "/../resources/hw10/instance_" + problems[problem_index] + ".mat"),
        tmp_dir,
        solvers[solver_index]
    );
    data_utils::RemoveDir(tmp_dir);
    return EXIT_SUCCESS;
}
