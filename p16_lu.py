import numpy as np


def lu(A, tol=1e-14):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    assert A.shape == (n, n), "lu() requires a square matrix"

    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        if abs(U[k, k]) < tol:
            raise ValueError(f"Zero pivot at position ({k},{k}).")
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return L, U


def solve_lu(L, U, b):
    b = np.array(b, dtype=float)
    n = len(b)

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x


def sherman_morrison_update(A_inv, u, v):
    A_inv = np.array(A_inv, dtype=float)
    u = np.array(u, dtype=float).reshape(-1, 1)
    v = np.array(v, dtype=float).reshape(1, -1)

    Au = A_inv @ u
    vA = v @ A_inv
    denom = 1.0 + (v @ Au).item()

    if abs(denom) < 1e-14:
        raise ValueError("Sherman-Morrison denominator is zero: updated matrix is singular")

    return A_inv - (Au @ vA) / denom


def test_lu_basic():
    A = np.array([[2.0, 1.0, 1.0],
                  [4.0, 3.0, 3.0],
                  [8.0, 7.0, 9.0]])
    L, U = lu(A)
    assert np.allclose(np.tril(L), L)
    assert np.allclose(np.diag(L), np.ones(3))
    assert np.allclose(np.triu(U), U)
    assert np.allclose(L @ U, A), f"L @ U != A:\n{L @ U}\nvs\n{A}"
    print("test_lu_basic passed")


def test_solve_lu():
    A = np.array([[3.0, 1.0, -1.0],
                  [6.0, 4.0,  1.0],
                  [3.0, 7.0,  5.0]])
    b = np.array([1.0, 2.0, 3.0])
    L, U = lu(A)
    x = solve_lu(L, U, b)
    assert np.allclose(A @ x, b, atol=1e-10), f"Ax != b: {A @ x} vs {b}"
    x_ref = np.linalg.solve(A, b)
    assert np.allclose(x, x_ref, atol=1e-10)
    print("test_solve_lu passed")


def test_sherman_morrison():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((4, 4))
    A += 5 * np.eye(4)
    u = rng.standard_normal(4)
    v = rng.standard_normal(4)

    A_inv = np.linalg.inv(A)
    B = A + np.outer(u, v)
    B_inv_sm = sherman_morrison_update(A_inv, u, v)
    B_inv_ref = np.linalg.inv(B)

    assert np.allclose(B_inv_sm, B_inv_ref, atol=1e-10), (
        f"Sherman-Morrison result differs from direct inverse:\n"
        f"max error = {np.max(np.abs(B_inv_sm - B_inv_ref))}"
    )
    print("test_sherman_morrison passed")


def test_identity_lu():
    n = 5
    A = np.eye(n)
    L, U = lu(A)
    assert np.allclose(L, np.eye(n))
    assert np.allclose(U, np.eye(n))
    print("test_identity_lu passed")


def test_multiple_rhs():
    A = np.array([[4.0, 3.0],
                  [6.0, 3.0]])
    L, U = lu(A)
    for b_vec in [np.array([10.0, 12.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]:
        x = solve_lu(L, U, b_vec)
        assert np.allclose(A @ x, b_vec, atol=1e-12)
    print("test_multiple_rhs passed")


if __name__ == "__main__":
    test_lu_basic()
    test_solve_lu()
    test_identity_lu()
    test_multiple_rhs()
    test_sherman_morrison()
    print("P16 OK")

    A = np.array([[2.0, -1.0,  0.0],
                  [-1.0, 2.0, -1.0],
                  [0.0, -1.0,  2.0]])
    L, U = lu(A)
    print("L:\n", np.round(L, 4))
    print("U:\n", np.round(U, 4))

    b = np.array([1.0, 0.0, 1.0])
    x = solve_lu(L, U, b)
    print("x:", np.round(x, 10))

    A2 = np.array([[3.0, 1.0], [1.0, 2.0]])
    u2, v2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    B2_inv = sherman_morrison_update(np.linalg.inv(A2), u2, v2)
    print("SM inv:\n", np.round(B2_inv, 6))
    print("direct:\n", np.round(np.linalg.inv(A2 + np.outer(u2, v2)), 6))
