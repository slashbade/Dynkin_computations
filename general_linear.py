import numpy as np
from basic import *

def bracket_m(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.dot(x, y) - np.dot(y, x)

def E_unit(n: int, i: int, j:int) -> np.ndarray:
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
        assert self.basis.shape[1:] == matrix.shape, print(self.basis.shape, matrix.shape)
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
                structure_constant[i, j, :] = self._as_coord(bracket_m(self.basis[i], self.basis[j]))
        
        return structure_constant
    

if __name__ == "__main__":
    # print(E_unit(4, 1, 3))
    basis = np.stack([E_unit(3, 0, 0) - E_unit(3, 1, 1),
                      E_unit(3, 1, 1) - E_unit(3, 2, 2),
                      E_unit(3, 0, 1),
                      E_unit(3, 0, 2),
                      E_unit(3, 1, 2),
                      E_unit(3, 1, 0),
                      E_unit(3, 2, 0),
                      E_unit(3, 2, 1)])

    GL = GeneralLinear(basis=basis)
    # print(L.basis.shape)
    # matrix = L._as_matrix(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    # coord = L._as_coord(matrix)
    
    x_mat = np.array([[1, 2, 3], 
                      [4, 50, 6],
                      [7, 8,-51]])
    
    y_mat = np.array([[2, 3, 4],
                      [5, 60, 7],
                      [8, 9,-62]])
    
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
    subspace_basis = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]
    )
    L_sub = sub_lie_algebra(L, subspace_basis)
    print(adjoint(L_sub, np.array([1, 200, 5, 1, 0, 0, 1, 4])))
    