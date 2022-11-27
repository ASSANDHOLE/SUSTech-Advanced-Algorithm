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
