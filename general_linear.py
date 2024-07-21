import numpy as np
from numpy.typing import NDArray
from basic import *


def bracket_m(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.dot(x, y) - np.dot(y, x)


def E_unit(n: int, i: int, j: int) -> np.ndarray:
    zeros_mat = np.zeros((n, n))
    zeros_mat[i, j] = 1
    return zeros_mat


class GeneralLinear:
    def __init__(self, basis: np.ndarray) -> None:
        """Use matrix basis to initialize a Lie algebra

        Args:
            basis (np.ndarray): basis tensor, shape (M, N, N)
        """
        self.basis = basis
        self.gram_matrix_inv = self._gram_matrix_inv()
        self.structure_constant = self._structure_constant()

    def _gram_matrix_inv(self) -> np.ndarray:
        """computes the inv of gram matrix of Lie algebra basis, for the transformation
        between coord and matrix

        Returns:
            np.ndarray: gram matrix inv
        """
        basis_num = self.basis.shape[0]
        gram_matrix = np.zeros((basis_num, basis_num))
        for i in range(basis_num):
            for j in range(basis_num):
                gram_matrix[i, j] = np.sum(self.basis[i] * self.basis[j])
        # print(gram_matrix)
        return np.linalg.inv(gram_matrix)

    def _as_matrix(self, coord: np.ndarray) -> np.ndarray:
        """Converts coord form to matrix form

        Args:
            coord (np.ndarray): coordinate with respect to basis

        Returns:
            np.ndarray: matrix in GLn
        """
        return np.sum(coord[:, np.newaxis, np.newaxis] * self.basis, axis=0)

    def _as_coord(self, matrix: np.ndarray) -> np.ndarray:
        """Converts matrix form to coord form

        Args:
            matrix (np.ndarray): matrix in GLn

        Returns:
            np.ndarray: coord with respect to basis
        """
        assert self.basis.shape[1:] == matrix.shape, print(
            self.basis.shape, matrix.shape
        )
        basis_num = self.basis.shape[0]
        b_inner = np.zeros(basis_num)
        for i in range(basis_num):
            b_inner[i] = np.sum(matrix * self.basis[i])
        coord = np.dot(self.gram_matrix_inv, b_inner)
        # print(self.gram_matrix_inv, b_inner)
        return coord

    def _structure_constant(self) -> np.ndarray:
        """Calculates structure constant for a Lie algebra

        Returns:
            np.ndarray: structure constant inthe shape (basis_num, basis_num, basis_num)
        """
        basis_num = self.basis.shape[0]
        structure_constant = np.zeros((basis_num, basis_num, basis_num))

        for i in range(basis_num):
            for j in range(basis_num):
                structure_constant[i, j, :] = self._as_coord(
                    bracket_m(self.basis[i], self.basis[j])
                )

        return structure_constant


def type_A_basis(rank: int) -> np.ndarray:
    """Generates basis for type A Lie algebra

    Args:
        rank (int): rank of the Lie algebra

    Returns:
        np.ndarray: basis for type A Lie algebra
    """
    basis_num = (rank + 1) * (rank + 1) - 1
    matrix_dim = rank + 1
    basis = np.zeros((basis_num, matrix_dim, matrix_dim))
    count = 0
    for i in range(matrix_dim - 1):
        basis[count] = E_unit(matrix_dim, i, i) - E_unit(matrix_dim, i + 1, i + 1)
        count += 1
    for i in range(matrix_dim):
        for j in range(i + 1, matrix_dim):
            basis[count] = E_unit(matrix_dim, i, j)
            count += 1
    for i in range(matrix_dim):
        for j in range(i):
            basis[count] = E_unit(matrix_dim, i, j)
            count += 1
    print(basis_num, count)
    return basis


def type_B_basis(rank: int) -> np.ndarray:
    """Generates basis for type B Lie algebra

    Args:
        rank (int): rank of the Lie algebra

    Returns:
        np.ndarray: basis for type B Lie algebra
    """
    basis_num = 2 * rank * rank + rank
    matrix_dim = rank * 2 + 1
    basis = np.zeros((basis_num, matrix_dim, matrix_dim))
    count = 0
    for i in range(1, rank + 1):
        basis[count] = E_unit(matrix_dim, i, i) - E_unit(matrix_dim, rank + i, rank + i)
        count += 1
    for i in range(rank):
        basis[count] = E_unit(matrix_dim, 0, rank + i + 1) - E_unit(
            matrix_dim, i + 1, 0
        )
        count += 1
    for i in range(rank):
        basis[count] = E_unit(matrix_dim, 0, i + 1) - E_unit(
            matrix_dim, rank + i + 1, 0
        )
        count += 1
    for i in range(rank):
        for j in range(rank):
            if i != j:
                basis[count] = E_unit(matrix_dim, i + 1, j + 1) - E_unit(
                    matrix_dim, rank + j + 1, rank + i + 1
                )
                count += 1
    for i in range(rank):
        for j in range(i + 1, rank):
            basis[count] = E_unit(matrix_dim, i + 1, rank + j + 1) - E_unit(
                matrix_dim, j + 1, rank + i + 1
            )
            count += 1
    for i in range(rank):
        for j in range(i):
            basis[count] = E_unit(matrix_dim, rank + i + 1, j + 1) - E_unit(
                matrix_dim, rank + j + 1, i + 1
            )
            count += 1

    return basis


def type_C_basis(rank: int) -> np.ndarray:
    """Generates basis for type C Lie algebra

    Args:
        rank (int): rank of the Lie algebra

    Returns:
        np.ndarray: basis for type C Lie algebra
    """
    basis_num = 2 * rank * rank + rank
    matrix_dim = rank * 2
    basis = np.zeros((basis_num, matrix_dim, matrix_dim))
    count = 0
    for i in range(rank):
        basis[count] = E_unit(matrix_dim, i, i) - E_unit(matrix_dim, rank + i, rank + i)
        count += 1
    for i in range(rank):
        for j in range(rank):
            if i != j:
                basis[count] = E_unit(matrix_dim, i, j) - E_unit(
                    matrix_dim, rank + j, rank + i
                )
                count += 1
    for i in range(rank):
        basis[count] = E_unit(matrix_dim, i, rank + i)
        count += 1
    for i in range(rank):
        for j in range(i + 1, rank):
            basis[count] = E_unit(matrix_dim, i, rank + j) + E_unit(
                matrix_dim, j, rank + i
            )
            count += 1

    for i in range(rank):
        basis[count] = E_unit(matrix_dim, rank + i, i)
        count += 1
    for i in range(rank):
        for j in range(i + 1, rank):
            basis[count] = E_unit(matrix_dim, rank + j, i) + E_unit(
                matrix_dim, rank + i, j
            )
            count += 1
    print(basis_num, count)
    return basis

def type_D_basis(rank: int) -> NDArray[np.float16]:
    basis_num = 2 * rank * rank - rank
    matrix_dim = rank * 2
    basis = np.zeros((basis_num, matrix_dim, matrix_dim))
    count = 0
    for i in range(rank):
        basis[count] = E_unit(matrix_dim, i, i) - E_unit(matrix_dim, rank + i, rank + i)
        count += 1
    for i in range(rank):
        for j in range(rank):
            if i != j:
                basis[count] = E_unit(matrix_dim, i, j) - E_unit(
                    matrix_dim, rank + j, rank + i
                )
                count += 1
    for i in range(rank):
        for j in range(i + 1, rank):
            basis[count] = E_unit(matrix_dim, i, rank + j) + E_unit(
                matrix_dim, j, rank + i
            )
            count += 1
    for i in range(rank):
        for j in range(i):
            basis[count] = E_unit(matrix_dim, rank + i, j) + E_unit(
                matrix_dim, rank + j, i
            )
            count += 1
    return basis

def type_gl_basis(rank: int) -> NDArray[np.float16]:
    basis_num = rank * rank
    matrix_dim = rank
    basis = np.zeros((basis_num, matrix_dim, matrix_dim))
    count = 0
    for i in range(rank):
        for j in range(rank):
            basis[count] = E_unit(matrix_dim, i, j)
            count += 1
    return basis

if __name__ == "__main__":
    # print(E_unit(4, 1, 3))
    basis = np.stack(
        [
            E_unit(3, 0, 0) - E_unit(3, 1, 1),
            E_unit(3, 1, 1) - E_unit(3, 2, 2),
            E_unit(3, 0, 1),
            E_unit(3, 0, 2),
            E_unit(3, 1, 2),
            E_unit(3, 1, 0),
            E_unit(3, 2, 0),
            E_unit(3, 2, 1),
        ]
    )

    GL = GeneralLinear(basis=basis)
    # print(L.basis.shape)
    # matrix = L._as_matrix(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    # coord = L._as_coord(matrix)

    x_mat = np.array([[1, 2, 3], [4, 50, 6], [7, 8, -51]])

    y_mat = np.array([[2, 3, 4], [5, 60, 7], [8, 9, -62]])

    coord1 = GL._as_coord(x_mat)
    # print(coord1)
    # matrix1 = L._as_matrix(coord1)
    # print(matrix1)
    # print(f"Matrix y as coordinate: {GL._as_coord(y_mat)}")
    # print(f"Bracket product of x and y {bracket(x_mat, y_mat)}")
    # print(f"Bracket product as coordinate: {GL._as_coord(bracket(x_mat, y_mat))}")

    structure_constant = GL.structure_constant
    L = LieAlgebra(structure_constant)
    print(adjoint(L, GL._as_coord(x_mat)))
    print(adjoint(L, np.array([1, 0, 0, 0, 0, 0, 0, 0])))
    print(bracket(L, GL._as_coord(x_mat), GL._as_coord(y_mat)))
    subspace_basis = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])
    L_sub = LieSubalgebra(L, subspace_basis)
    print(L_sub.structure_constant)
    print(adjoint(L_sub, np.array([1, 200, 0, 0, 0, 0, 0, 0])))
    # print(Killing_form_basis_matrix(L))
    # print(type_A_basis(3)[14])
    GL1 = GeneralLinear(type_C_basis(5))
