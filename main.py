import time
import numpy as np
from scipy.io import mmread
from library.jacobi_method import jacobi
from library.gauss_seidel_method import gauss_seidel
from library.gradient_method import gradient
from library.conjugate_gradient_method import conjugate_gradient
#from library.plot import plot_results

matrix_files = ["spa1.mtx", "spa2.mtx", "vem1.mtx", "vem2.mtx"]
tols = [1e-4, 1e-6, 1e-8, 1e-10]
solvers = {
    "Jacobi": jacobi,
    "Gauss-Seidel": gauss_seidel,
    "Gradient": gradient,
    "Conjugate Gradient": conjugate_gradient
}

all_results = {}

for file in matrix_files:
    print(f"\n==== Matrix: {file} ====")
    A = mmread(f"dati/{file}").toarray()
    x_exact = np.ones(A.shape[0])
    b = A @ x_exact
    matrix_results = {}

    for tol in tols:
        print(f"\nTOL = {tol}")
        method_results = {}
        for name, method in solvers.items():
            start = time.time()
            x_approx, iters = method(A, b, tol=tol)
            elapsed = time.time() - start
            error = np.linalg.norm(x_exact - x_approx) / np.linalg.norm(x_exact)
            print(f"{name:20} | iter: {iters:5d} | error: {error:.2e} | time: {elapsed:.4f}s")
            method_results[name] = (iters, error, elapsed)
        matrix_results[tol] = method_results
    all_results[file] = matrix_results

# Salvataggio e plot finale
data = {
    'results': all_results,
    'tols': tols,
    'solvers': list(solvers.keys()),
    'matrices': matrix_files
}
#plot_results(data)
