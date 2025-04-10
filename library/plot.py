import matplotlib.pyplot as plt
import os

def plot_results(data):
    os.makedirs("plots", exist_ok=True)

    for matrix in data['matrices']:
        results = data['results'][matrix]

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
            plt.title(f"{matrix} - {metric.title()}")
            plt.legend()
            plt.grid(True, which='both', ls='--', lw=0.5)
            plt.tight_layout()
            matrix_dir = f"plots/{matrix}"
            os.makedirs(matrix_dir, exist_ok=True)
            plt.savefig(f"{matrix_dir}/{metric}.png")
            plt.show()
            plt.close()