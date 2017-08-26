import { OpInput, OpOutput, Scalar } from '../../commonTypes';
import { DType } from '../../dtype';
import { Tensor } from '../../tensor';

export const enum MatrixModifier {
    None = 0,
    Transposed = 1,
    Hermitian = 2
}

export interface IMatrixOpProvider {

    /**
     * Checks if a matrix is (skew-)symmetric.
     * @param x Input matrix.
     * @param skew (Optional) If set to true, will check if the input is skew-
     *             symmetric. Default value is false.
     */
    isSymmetric(x: OpInput, skew?: boolean): boolean;

    /**
     * Check if a matrix is (skew-)Hermitian.
     * Note: if the input matrix is real, this function is equivalent to
     *       isSymmetric().
     * @param x Input matrix.
     * @param skew (Optional) If set to true, will check if the input is skew-
     *             Hermitian. Default value is false.
     */
    isHermitian(x: OpInput, skew?: boolean): boolean;

    /**
     * Creates a matrix with the element on the main diagonal set to one.
     * @param m Number of rows.
     * @param n (Optional) Number of columns. Default value is m.
     * @param dtype (Optional) Data type. Default value is FLOAT64.
     */
    eye(m: number, n?: number, dtype?: DType): Tensor;

    /**
     * Creates a Hilbert matrix.
     * @param m Number of rows/columns.
     */
    hilb(n: number): Tensor;

    /**
     * Creates a diagonal matrix or extracts diagonal elements.
     * If the input is a 1D vector (ndim = 1), its elements will be used to
     * create a diagonal matrix.
     * If the input is a m x n matrix (ndim = 2), its diagonal elements will be
     * extracted and a 1D vector of length min(m, n) containing the diagonal
     * elements will be returned.
     * @param x Matrix/vector input.
     * @param k (Optional) If specified, will operate on the k-th diagonal.
     *          For upper diagonals, k > 0; for lower diagonals, k < 0; for the
     *          main diagonal, k = 0. Default value is 0.
     */
    diag(x: OpInput, k?: number): Tensor;

    /**
     * Create a Vandermonde matrix.
     * @param x An 1D vector specifying the first column of the matrix.
     * @param n (Optional) Number of columns in the output. Default value is
     *          the length of x.
     * @param increasing (Optional) If set to true, the powers increase from
     *                   left to right. If set to false, the powers increase
     *                   from right to left. Default value is false.
     */
    vander(x: OpInput, n?: number, increasing?: boolean): Tensor;

    /**
     * Extracts the lower triangular part of the input matrix as a new matrix.
     * @param x Input matrix.
     * @param k (Optional) Offset. For upper diagonals, k > 0; for lower
     *          diagonals, k < 0; for the main diagonal, k = 0.
     *          Default value is 0.
     */
    tril(x: OpInput, k?: number): Tensor;

    /**
     * Extracts the upper triangular part of the input matrix as a new matrix.
     * @param x Input matrix.
     * @param k (Optional) Offset. For upper diagonals, k > 0; for lower
     *          diagonals, k < 0; for the main diagonal, k = 0.
     *          Default value is 0.
     */
    triu(x: OpInput, k?: number): Tensor;

    /**
     * Performs matrix multiplication.
     * @param x Input matrix x.
     * @param y Input matrix y.
     * @param yModifier Specifies whether whether transpose or Hermitian
     *                  operation needs to be applied to y before performing the
     *                  matrix multiplication. For example, if `yModifier` is
     *                  set to `MM_TRANSPOSED`, this function will compute
     *                  x y^T.
     */
    matmul(x: OpInput, y: OpInput, yModifier?: MatrixModifier): OpOutput;

    /**
     * Computes the Kronecker product between two matrices.
     * @param x Input matrix x.
     * @param y Input matrix y.
     */
    kron(x: OpInput, y: OpInput): Tensor;

    /**
     * Gets the transpose of the input matrix.
     * @param x Input matrix.
     */
    transpose(x: OpInput): Tensor;

    /**
     * Gets the Hermitian of the input matrix.
     * @param x Input matrix.
     */
    hermitian(x: OpInput): Tensor;

    /**
     * Computes the trace of the input matrix (must be square).
     * @param x Input matrix.
     */
    trace(x: OpInput): Scalar;

    /**
     * Computes the inverse of the input matrix (must be square).
     * Note: this function uses PLU decomposition to compute the inverse.
     * @param x Input matrix.
     */
    inv(x: OpInput): Tensor;

    /**
     * Computes the determinant of the input matrix (must be square).
     * Note: this function uses LUP decomposition to compute the inverse.
     * @param x Input matrix.
     */
    det(x: OpInput): Scalar;

    /**
     * Computes matrix/vector norms.
     * For vectors,
     * 1) if p = 0, counts the number of non-zero elements;
     * 2) if p > 0 (including Infinity), computes the L-p norm of the vector.
     * For matrices,
     * 1) if p = 1, 2, or Infinity, computes the induced matrix norm;
     * 2) if p = 'fro', computes the Frobenius norm.
     * @param x Input matrix/vector.
     * @param p Determines which norm will be calculated. For vectors, p must be
     *          nonnegative. For matrices, p can only be 1, 2, Infinity or
     *          'fro'.
     */
    norm(x: OpInput, p: number | 'fro'): number;

    /**
     * Performs LUP decomposition and return the results in the compact form.
     * @param x Input matrix.
     */
    lu(x: OpInput, compact: true): [Tensor, number[]];
    /**
     * Performs LUP decomposition and return the results in the full form.
     * @param x Input matrix.
     */
    lu(x: OpInput, compact: false): [Tensor, Tensor, Tensor];

    /**
     * Computes the singular value decomposition.
     * Returns a 3-item tuple [U, S, V] such that x = USV^H.
     * Let the shape of x be m x n, then the shapes of U, S, V, are
     * m x min(m,n), min(m,n) x n, n x n, respectively. 
     * @param x Input matrix x.
     * @returns A 3-item tuple [U, S, V] such that x = USV^H.
     */
    svd(x: OpInput): [Tensor, Tensor, Tensor];

    /**
     * Estimates the rank of the input matrix via from its singular values.
     * @param x Input matrix.
     * @param tol Tolerance for small singular values.
     */
    rank(x: OpInput, tol?: number): number;

    /**
     * Estimates the condition number of the input matrix from its singular
     * values.
     * @param x Input matrix.
     */
    cond(x: OpInput): number;

    /**
     * Obtains the pseudo inverse of the input matrix from its singular value
     * decomposition.
     * @param x Input matrix.
     * @param tol Tolerance for small singular values.
     */
    pinv(x: OpInput, tol?: number): Tensor;

    /**
     * Computes the eigendecomposition of the input matrix.
     * Returns a 2-item tuple [E, L] such that x E = E L.
     * If the input matrix is symmetric/Hermitian, E will be unitary. For
     * general matrices E is unnormalized.
     * @param x Input matrix.
     */
    eig(x: OpInput): [Tensor, Tensor];
    /**
     * Computes the eigendecomposition of the input matrix.
     * Returns a 2-item tuple [E, L] such that x E = E L.
     * If the input matrix is symmetric/Hermitian, E will be unitary. For
     * general matrices E is unnormalized.
     * @param x Input matrix.
     */
    eig(x: OpInput, evOnly: false): [Tensor, Tensor];
    /**
     * Computes the eigenvalues. Returns a vector of eigenvalues.
     * @param x Input matrix.
     */
    eig(x: OpInput, evOnly: true): Tensor;


    /**
     * Computes the Cholesky decomposition of the input matrix.
     * Note: this function assumes the input is symmetric/Hermitian and only
     *       uses the lower triangular part of the input matrix. You are
     *       responsible for ensuring that the input matrix is symmetric or
     *       Hermitian.
     * @param x Input matrix.
     * @returns A lower triangular matrix L such that L*L' produces the original
     *          matrix.
     * @throws Throws an error when the input matrix is not positive definite. 
     */
    chol(x: OpInput): Tensor;

    /**
     * Computes the QR decomposition of the input matrix X.
     * Returns a tuple [Q, R, P] such that XP = QR.
     * @param x Input matrix.
     */
    qr(x: OpInput): [Tensor, Tensor, Tensor];

    /**
     * Solves the linear system AX = B, where A: m x n, X: n x p, B: m x p.
     * If m = n, LUP decomposition is used to obtain X.
     * If m > n, column pivoted QR decomposition is used to obtain a least
     * square solution.
     * If m < n or A is rank deficient, the solution cannot be trusted.
     * @param a Matrix A.
     * @param b Matrix B. 
     */
    linsolve(a: OpInput, b: OpInput): Tensor;

}
