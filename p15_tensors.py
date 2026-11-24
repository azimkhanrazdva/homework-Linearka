import numpy as np


def rank_one(u, v, w):
    return np.einsum("i,j,k->ijk", u, v, w)


def dot_product(u, v):
    return np.einsum("i,i->", u, v)


def matrix_vector(A, v):
    return np.einsum("ij,j->i", A, v)


def matrix_matrix(A, B):
    return np.einsum("ij,jk->ik", A, B)


def outer_product(u, v):
    return np.einsum("i,j->ij", u, v)


def trace(A):
    return np.einsum("ii->", A)


def frobenius_inner(A, B):
    return np.einsum("ij,ij->", A, B)


def transpose(A):
    return np.einsum("ij->ji", A)


def contract_last_first(A, B):
    return np.einsum("...i,ij->...j", A, B)


def batch_matmul(A, B):
    return np.einsum("bij,bjk->bik", A, B)


def kron_via_einsum(A, B):
    m, n = A.shape
    p, q = B.shape
    T = np.einsum("ij,kl->ikjl", A, B)
    return T.reshape(m * p, n * q)


def tensor_reshape_demo():
    T = np.arange(24).reshape(2, 3, 4)
    print("Original shape:", T.shape)
    print("As (6, 4):\n", T.reshape(6, 4))
    print("As (2, 12):\n", T.reshape(2, 12))
    print("As (24,):\n", T.reshape(24))
    return T


def test_rank_one():
    u = np.array([1.0, 2.0])
    v = np.array([3.0, 4.0, 5.0])
    w = np.array([6.0, 7.0])
    T = rank_one(u, v, w)
    assert T.shape == (2, 3, 2)
    assert np.isclose(T[1, 2, 0], 60.0)
    print("test_rank_one passed")


def test_einsum_ops():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    B = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    assert np.isclose(dot_product(u, v), np.dot(u, v))
    assert np.allclose(matrix_vector(A, np.array([1.0, 2.0])), A @ np.array([1.0, 2.0]))
    assert np.allclose(matrix_matrix(A, B), A @ B)
    assert np.allclose(outer_product(u[:2], v[:2]), np.outer(u[:2], v[:2]))
    assert np.isclose(trace(np.eye(5)), 5.0)

    C = np.random.randn(3, 4)
    D = np.random.randn(3, 4)
    assert np.isclose(frobenius_inner(C, D), np.sum(C * D))
    assert np.allclose(transpose(A), A.T)
    print("test_einsum_ops passed")


def test_kron():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[0.0, 5.0], [6.0, 7.0]])
    result = kron_via_einsum(A, B)
    expected = np.kron(A, B)
    assert np.allclose(result, expected), f"Kron mismatch:\n{result}\nvs\n{expected}"
    print("test_kron passed")


def test_batch_matmul():
    rng = np.random.default_rng(42)
    A = rng.standard_normal((5, 3, 4))
    B = rng.standard_normal((5, 4, 2))
    result = batch_matmul(A, B)
    expected = np.stack([A[i] @ B[i] for i in range(5)])
    assert np.allclose(result, expected)
    print("test_batch_matmul passed")


if __name__ == "__main__":
    test_rank_one()
    test_einsum_ops()
    test_kron()
    test_batch_matmul()
    print("P15 OK")

    tensor_reshape_demo()

    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0])
    w = np.array([6.0, 7.0, 8.0, 9.0])
    T = rank_one(u, v, w)
    print("rank_one shape:", T.shape)
    print("T[0,0,:]:", T[0, 0, :])

    A = np.eye(2)
    B = np.array([[1, 2], [3, 4]], dtype=float)
    print("I kron B =\n", kron_via_einsum(A, B))
