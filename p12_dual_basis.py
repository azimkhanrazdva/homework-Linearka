import numpy as np
import matplotlib.pyplot as plt


def dual_basis(P):
    return np.linalg.inv(P).T


def lagrange_basis_poly(x_nodes, j):
    nodes = list(x_nodes)
    xj = nodes[j]
    poly = np.array([1.0])
    denom = 1.0
    for k, xk in enumerate(nodes):
        if k == j:
            continue
        poly = np.polymul(poly, [1.0, -xk])
        denom *= (xj - xk)
    return poly / denom


def lagrange_interpolate(x_nodes, y_values, x_eval):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    result = np.zeros_like(x_eval, dtype=float)
    for j in range(len(x_nodes)):
        Lj = np.ones_like(x_eval, dtype=float)
        for k in range(len(x_nodes)):
            if k == j:
                continue
            Lj *= (x_eval - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
        result += y_values[j] * Lj
    return result


def runge_function(x):
    return 1.0 / (1.0 + 25.0 * x ** 2)


def demo_runge(save_path=None):
    x_fine = np.linspace(-1, 1, 500)
    y_true = runge_function(x_fine)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ns = [5, 10, 15]

    for ax, n in zip(axes, ns):
        x_nodes = np.linspace(-1, 1, n)
        y_nodes = runge_function(x_nodes)
        y_interp = lagrange_interpolate(x_nodes, y_nodes, x_fine)

        ax.plot(x_fine, y_true, "k-", lw=1.5, label="f(x)")
        ax.plot(x_fine, y_interp, "b--", lw=1.5, label=f"P_{n-1}(x)")
        ax.plot(x_nodes, y_nodes, "ro", ms=4)
        ax.set_ylim(-2, 2)
        ax.set_title(f"n = {n} equally spaced nodes")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Runge's phenomenon: Lagrange interpolation of 1/(1+25x²)", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def vandermonde_interpolate(x_nodes, y_values, x_eval):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    n = len(x_nodes)
    V = np.vander(x_nodes, increasing=True)
    coeffs = np.linalg.solve(V, y_values)
    x_eval = np.asarray(x_eval, dtype=float)
    result = np.zeros_like(x_eval)
    for j, c in enumerate(coeffs):
        result += c * x_eval ** j
    return result


def test_dual_basis():
    P = np.array([[1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1]], dtype=float)
    Q = dual_basis(P)
    product = Q.T @ P
    assert np.allclose(product, np.eye(3), atol=1e-10), f"Q.T @ P != I:\n{product}"
    print("test_dual_basis passed")


def test_lagrange_exact():
    x_nodes = np.array([-1.0, 0.0, 1.0])
    y_nodes = x_nodes ** 2
    x_eval = np.linspace(-1, 1, 50)
    y_interp = lagrange_interpolate(x_nodes, y_nodes, x_eval)
    assert np.allclose(y_interp, x_eval ** 2, atol=1e-12), "Lagrange failed on x^2"
    print("test_lagrange_exact passed")


def test_vandermonde_vs_lagrange():
    x_nodes = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y_nodes = np.sin(x_nodes)
    x_eval = np.linspace(-2, 2, 100)
    y_lag = lagrange_interpolate(x_nodes, y_nodes, x_eval)
    y_van = vandermonde_interpolate(x_nodes, y_nodes, x_eval)
    assert np.allclose(y_lag, y_van, atol=1e-9), "Lagrange and Vandermonde differ"
    print("test_vandermonde_vs_lagrange passed")


if __name__ == "__main__":
    test_dual_basis()
    test_lagrange_exact()
    test_vandermonde_vs_lagrange()
    print("P12 OK")

    P = np.array([[2, 1], [1, 3]], dtype=float)
    Q = dual_basis(P)
    print("Q.T @ P =\n", np.round(Q.T @ P, 10))

    demo_runge(save_path=r"C:\Users\sayo\Desktop\hwla\runge_phenomenon.png")
