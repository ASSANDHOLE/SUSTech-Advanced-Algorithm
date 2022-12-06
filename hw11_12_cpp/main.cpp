#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <ortools/linear_solver/linear_solver.h>

extern "C" double Hw11MipSolver(const uint8_t *A, int a_sz0, int a_sz1, const float *w, int *x) {
    // a_sz0 == w_sz == x_sz == number of variables (vertices)
    // a_sz1 == number of edges
    using namespace operations_research;
    std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
    if (solver == nullptr) {
        throw std::runtime_error("Could not create solver SCIP");
    }

    const double lower_bound = 0.0;
    const double upper_bound = 1.0;
    const double infinity = solver->infinity();

    std::vector<MPVariable const *> x_vars;
    x_vars.reserve(a_sz0);
    for (int i = 0; i < a_sz0; ++i) {
        x_vars.push_back(solver->MakeIntVar(lower_bound, upper_bound, "x_" + std::to_string(i)));
    }

    for (int i = 0; i < a_sz1; ++i) {
        MPConstraint * const con = solver->MakeRowConstraint(1, infinity, "c_" + std::to_string(i));
        for (int j = 0; j < a_sz0; ++j) {
            if (A[i * a_sz0 + j]) {
                con->SetCoefficient(x_vars[j], 1);
            }
        }
    }

    MPObjective * const objective = solver->MutableObjective();
    for (int i = 0; i < a_sz0; ++i) {
        objective->SetCoefficient(x_vars[i], w[i]);
    }
    objective->SetMinimization();

    const auto result_status = solver->Solve();
    if (result_status != MPSolver::OPTIMAL) {
        throw std::runtime_error("Solver did not find optimal solution");
    }
    for (int i = 0; i < a_sz0; ++i) {
        x[i] = int(x_vars[i]->solution_value());
    }
    return objective->Value();
}

extern "C" double Hw11LpBasedSolver(const uint8_t *A, int a_sz0, int a_sz1, const float *w, float *x) {
    // a_sz0 == w_sz == x_sz == number of variables (vertices)
    // a_sz1 == number of edges
    using namespace operations_research;
    std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
    if (solver == nullptr) {
        throw std::runtime_error("Could not create solver SCIP");
    }

    const double lower_bound = 0.0;
    const double upper_bound = 1.0;
    const double infinity = solver->infinity();

    std::vector<MPVariable const *> x_vars;
    x_vars.reserve(a_sz0);
    for (int i = 0; i < a_sz0; ++i) {
        x_vars.push_back(solver->MakeNumVar(lower_bound, upper_bound, "x_" + std::to_string(i)));
    }

    for (int i = 0; i < a_sz1; ++i) {
        MPConstraint * const con = solver->MakeRowConstraint(1, infinity, "c_" + std::to_string(i));
        for (int j = 0; j < a_sz0; ++j) {
            if (A[i * a_sz0 + j]) {
                con->SetCoefficient(x_vars[j], 1);
            }
        }
    }

    MPObjective * const objective = solver->MutableObjective();
    for (int i = 0; i < a_sz0; ++i) {
        objective->SetCoefficient(x_vars[i], w[i]);
    }
    objective->SetMinimization();

    const auto result_status = solver->Solve();
    if (result_status != MPSolver::OPTIMAL) {
        throw std::runtime_error("Solver did not find optimal solution");
    }
    for (int i = 0; i < a_sz0; ++i) {
        x[i] = float(x_vars[i]->solution_value());
    }
    return objective->Value();
}

extern "C" double Hw12LpBasedSolver(const float *jobs_weight, const int jobs_len, const int *machines_2d_vec, const int *machines_shape, const int machines_num, float *x_ijs, const char *solver_name) {
    // jobs_weight: 1D array of length jobs_len
    // machines_2d_vec: 1D array of length sum(machines_shape), e.g., {{0, 1, 3}, {2, 3}, {1, 2}}.flatten()
    // machines_shape: 1D array of length machines_num
    // x_ijs: 1D array of length machines_len * jobs_len
    using namespace operations_research;
    std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver(solver_name));
    if (solver == nullptr) {
        throw std::runtime_error("Could not create solver" + std::string(solver_name));
    }
    std::vector<std::vector<int>> machines(machines_num);
    int cur = 0;
    for (int i = 0; i < machines_num; ++i) {
        machines[i].resize(machines_shape[i]);
        for (int j = 0; j < machines_shape[i]; ++j) {
            machines[i][j] = machines_2d_vec[cur++];
        }
    }
    // initialize variables
    std::vector<MPVariable const *> x_ijs_vars;
    const double infinity = solver->infinity();
    double upper;
    x_ijs_vars.reserve(machines_num * jobs_len);
    for (int i = 0; i < machines_num; ++i) {
        for (int j = 0; j < jobs_len; ++j) {
            if (std::find(machines[i].begin(), machines[i].end(), j) != machines[i].end()) {
                upper = infinity;
            } else {
                upper = 0.0;
            }
            x_ijs_vars.push_back(solver->MakeNumVar(0.0, upper, "x_" + std::to_string(i) + "_" + std::to_string(j)));
        }
    }

    // add equality constraints, i.e., sum_i(x_ijs) == jobs_weight[j]
    for (int j = 0; j < jobs_len; ++j) {
        MPConstraint * const con = solver->MakeRowConstraint(jobs_weight[j], jobs_weight[j], "c_eq_" + std::to_string(j));
        for (int i = 0; i < machines_num; ++i) {
            con->SetCoefficient(x_ijs_vars[i * jobs_len + j], 1);
        }
    }

    // set dummy minimization, i.e., minimize max(sum_j(x_ijs))
    MPVariable const * const dummy_var = solver->MakeNumVar(0.0, infinity, "x_dummy");
    for (int i = 0; i < machines_num; ++i) {
        MPConstraint * const con = solver->MakeRowConstraint(0.0, infinity, "c_dummy_" + std::to_string(i));
        for (int j = 0; j < jobs_len; ++j) {
            con->SetCoefficient(x_ijs_vars[i * jobs_len + j], -1);
        }
        con->SetCoefficient(dummy_var, 1);
    }

    MPObjective * const objective = solver->MutableObjective();
    objective->SetCoefficient(dummy_var, 1);
    objective->SetMinimization();

    const auto result_status = solver->Solve();
    if (result_status != MPSolver::OPTIMAL) {
        throw std::runtime_error("Solver did not find optimal solution");
    }
    cur = 0;
    for (const auto & var : x_ijs_vars) {
        x_ijs[cur++] = float(var->solution_value());
    }
    return objective->Value();
}
