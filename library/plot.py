import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_results(data):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("plots", now)
    os.makedirs(base_dir, exist_ok=True)

    for matrix in data['matrices']:
        results = data['results'][matrix]
        matrix_name = os.path.splitext(matrix)[0]
        matrix_dir = os.path.join(base_dir, matrix_name)
        os.makedirs(matrix_dir, exist_ok=True)

        for metric_idx, metric in enumerate(['iters', 'error']):
            plt.figure(figsize=(8, 6))
            for solver in data['solvers']:
                y = [
                    results[tol][solver][metric_idx]
                    for tol in data['tols']
                ]
                plt.plot(data['tols'], y, marker='o', label=solver)

            plt.xscale('log')
            if metric == 'iters':
                plt.yscale('linear')
                plt.ylabel('Numero di Iterazioni')
            else:
                plt.yscale('log')
                plt.ylabel('Errore Relativo')

            plt.xlabel('Tolleranza')
            plt.title(f"{matrix_name} - {metric.title()}")
            plt.legend()
            plt.grid(True, which='both', ls='--', lw=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(matrix_dir, f"{metric}.png"))
            plt.show()
            plt.close()
