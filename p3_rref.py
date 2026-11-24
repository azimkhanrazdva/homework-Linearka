import numpy as np


def rref(A, tol=1e-10):
    R = np.array(A, dtype=float)
    m, n = R.shape
    pivot_row = 0
    pivots = []

    for col in range(n):
        sub = np.abs(R[pivot_row:, col])
        best = np.argmax(sub) + pivot_row

        if R[best, col] < tol:
            continue

        R[[pivot_row, best]] = R[[best, pivot_row]]
        R[pivot_row] = R[pivot_row] / R[pivot_row, col]

        for row in range(m):
            if row != pivot_row and abs(R[row, col]) > tol:
                R[row] -= R[row, col] * R[pivot_row]

        pivots.append(col)
        pivot_row += 1
        if pivot_row == m:
            break

    R[np.abs(R) < tol] = 0.0
    return R, pivots


def rank(A):
    _, pivots = rref(A)
    return len(pivots)


def null_space_basis(A, tol=1e-10):
    m, n = A.shape
    R, pivots = rref(A, tol)
    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]

    basis = []
    for fc in free_cols:
        vec = np.zeros(n)
        vec[fc] = 1.0
        for i, pc in enumerate(pivots):
            if pc < n:
                vec[pc] = -R[i, fc]
        basis.append(vec)

    if basis:
        return np.column_stack(basis)
    else:
        return np.zeros((n, 0))


def col_space_basis(A):
    _, pivots = rref(A)
    return A[:, pivots]


def test_rref_basic():
    A = np.array([[2, 4, -2],
                  [4, 9, -3],
                  [-2, -3, 7]], dtype=float)
    R, pivots = rref(A)
    expected = np.eye(3)
    assert np.allclose(R, expected, atol=1e-9), f"Expected identity, got\n{R}"
    assert pivots == [0, 1, 2]
    print("test_rref_basic passed")


def test_rank_deficient():
    A = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [0, 1, 1]], dtype=float)
    r = rank(A)
    assert r == 2, f"Expected rank 2, got {r}"
    print("test_rank_deficient passed")


def test_null_space():
    A = np.array([[1, 2, 3],
                  [2, 4, 6]], dtype=float)
    N = null_space_basis(A)
    for j in range(N.shape[1]):
        residual = np.linalg.norm(A @ N[:, j])
        assert residual < 1e-9, f"Null space vector {j} has residual {residual}"
    assert N.shape[1] == 2, f"Expected nullity 2, got {N.shape[1]}"
    print("test_null_space passed")


def test_col_space():
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=float)
    C = col_space_basis(A)
    assert C.shape[1] == 2, f"Expected 2 basis vectors, got {C.shape[1]}"
    for j in range(A.shape[1]):
        coeffs, _, _, _ = np.linalg.lstsq(C, A[:, j], rcond=None)
        residual = np.linalg.norm(C @ coeffs - A[:, j])
        assert residual < 1e-9, f"Column {j} not in column space"
    print("test_col_space passed")


def test_identity_rref():
    A = np.eye(4)
    R, pivots = rref(A)
    assert np.allclose(R, A), "rref of identity should be identity"
    assert pivots == [0, 1, 2, 3]
    print("test_identity_rref passed")


def test_zero_matrix():
    A = np.zeros((3, 4))
    R, pivots = rref(A)
    assert pivots == [], "zero matrix has no pivots"
    assert np.allclose(R, A)
    print("test_zero_matrix passed")


if __name__ == "__main__":
    test_rref_basic()
    test_rank_deficient()
    test_null_space()
    test_col_space()
    test_identity_rref()
    test_zero_matrix()
    print("P3 OK")

    A = np.array([[1, 2, 0, -1],
                  [2, 4, 1,  0],
                  [0, 0, 1,  2]], dtype=float)
    R, piv = rref(A)
    print("RREF:\n", R)
    print("pivots:", piv, " rank:", rank(A))
    print("null:\n", null_space_basis(A))
    print("col:\n", col_space_basis(A))
