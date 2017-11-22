import { DataBlock } from '../commonTypes';
import { MatrixModifier } from './modifiers';

export interface IBlaoBackend {

    /**
     * Swaps the first n elements in A and B:
     *  A <-> B.
     * @param n Number of elements.
     * @param A Array A.
     * @param B Array B.
     */
    swap(n: number, A: DataBlock, B: DataBlock): void;

    /**
     * Scales the elements in A:
     *  A <- alpha * A.
     */
    scale(alpha: number, A: DataBlock): void;
    
    /**
     * Scales the elements in A (complex version):
     *  A <- alpha * A.
     */
    cscale(reAlpha: number, imAlpha: number, reA: DataBlock, imA: DataBlock): void;

    /**
     * Obtains the transpose of A:
     *  B <- A^T.
     */
    transpose(m: number, n: number, A: ArrayLike<number>, B: DataBlock): void;
    
    /**
     * Obtains the Hermitian of A:
     *  B <- A^H.
     */
    hermitian(m: number, n: number, reA: ArrayLike<number>, imA: ArrayLike<number>,
              reB: DataBlock, imB: DataBlock): void;

    /**
     * Performs rank-one update on A:
     *  A <- alpha * x * y^T + A.
     */
    ger(alpha: number, x: ArrayLike<number>, y: ArrayLike<number>, A: DataBlock): void;

    /**
     * Performs rank-one update on A (complex version without conjugate):
     *  A <- alpha * x * y^T + A.
     */
    cgeru(reAlpha: number, imAlpha: number, reX: ArrayLike<number>, imX: ArrayLike<number>,
          reY: ArrayLike<number>, imY: ArrayLike<number>, reA: DataBlock, imA: DataBlock): void;

    /**
     * Performs rank-one update on A (complex version):
     *  A <- alpha * x * y^H + A.
     */
    cgerc(reAlpha: number, imAlpha: number, reX: ArrayLike<number>, imX: ArrayLike<number>,
          reY: ArrayLike<number>, imY: ArrayLike<number>, reA: DataBlock, imA: DataBlock): void;

    /**
     * General matrix-vector multiplication:
     *  y <- alpha * M(A) * x + beta * y,
     * where M(A) = A, A^T, or A^H.
     */
    gemv(m: number, n: number, alpha: number, A: ArrayLike<number>, modA: MatrixModifier,
          x: ArrayLike<number>, beta: number, y: DataBlock): void;
    
    /**
     * General matrix-vector multiplication (complex version):
     *  y <- alpha * M(A) * x + beta * y,
     * where M(A) = A, A^T, or A^H.
     */
    cgemv(m: number, n: number, reAlpha: number, imAlpha: number, reA: ArrayLike<number>, imA: ArrayLike<number>,
          modA: MatrixModifier, reX: ArrayLike<number>, imX: ArrayLike<number>,
          reBeta: number, imBeta: number, reY: DataBlock, imY: DataBlock): void;

    /**
     * General matrix-matrix multiplication:
     *  C <- alpha * A * M(B) + beta * C,
     * where M(B) = B, B^T, or B^H.
     */
    gemm(m: number, n: number, k: number, alpha: number, A: ArrayLike<number>,
          B: ArrayLike<number>, modB: MatrixModifier, beta: number, C: DataBlock): void;

    /**
     * General matrix-matrix multiplication (complex version):
     *  C <- alpha * A * M(B) + beta * C,
     * where M(B) = B, B^T, or B^H.
     */
    cgemm(m: number, n: number, k: number, reAlpha: number, imAlpha: number,
          reA: ArrayLike<number>, imA: ArrayLike<number>, reB: ArrayLike<number>, imB: ArrayLike<number>,
          modB: MatrixModifier, reBeta: number, imBeta: number, reC: DataBlock, imC: DataBlock): void;

}

/**
 * Backend for LUP decomposition.
 */
export interface ILUBackend {

    /**
     * In-place LUP decomposition.
     * @param m Dimension of the matrix.
     * @param reX (Input/Output) Real part. Will be overwritten with the LU
     *            decomposition.
     * @param p (Output) Array storing the permutation vector. For instance, if
     *          p = [2, 0, 1], then the 1st row now stores the original 3rd row.
     * @returns Induced sign of the permutation (det(P)).
     */
    lu(m: number, reX: DataBlock, p: DataBlock): number;

    /**
     * In-place complex LUP decomposition.
     * @param m Dimension of the matrix.
     * @param reX (Input/Output) Real part. Will be overwritten with the real
     *            part of the LU decomposition.
     * @param imX (Input/Output) Imaginary part. Will be overwritten with the
     *            imaginary part of the LU decomposition.
     * @param p (Output) Array storing the permutation vector. For instance, if
     *          p = [2, 0, 1], then the 1st row now stores the original 3rd row.
     * @returns Induced sign of the permutation (det(P)).
     */
    clu(m: number, reX: DataBlock, imX: DataBlock, p: DataBlock): number;

    /**
     * Solves P*L*U*X = B, where P, L, U are m x m, and B is m x n.
     * B will be overwritten with the solution X.
     * @param m Number of rows in B.
     * @param n Number of columns in B.
     * @param reLU (Input) Compact storage of L and U of the PLU decomposition.
     * @param p (Input) Permutation vector.
     * @param reB (Output) Matrix B.
     */
    luSolve(m: number, n: number, reLU: ArrayLike<number>, p: ArrayLike<number>, reB: DataBlock): void;

    /**
     * Solves P*L*U*X = B (complex case), where P, L, U are m x m, and B is
     * m x n. B will be overwritten with the solution X.
     * @param m Number of rows in B.
     * @param n Number of columns in B.
     * @param reLU (Input) Real part of the compact storage of L and U of the
     *  PLU decomposition.
     * @param imLU (Input) Imaginary part of the compact storage of L and U of 
     *  the PLU decomposition.
     * @param p (Input) Permutation vector.
     * @param reB (Output) Real part of B.
     * @param imB (Output) Imaginary part of B.
     */
    cluSolve(m: number, n: number, reLU: ArrayLike<number>, imLU: ArrayLike<number>,
             p: ArrayLike<number>, reB: DataBlock, imB: DataBlock): void;

    /**
     * Converts a compact LU matrix to full L and U matrices.
     * @param m Dimension of the matrix.
     * @param isImaginaryPart Whether we are converting the imaginary part.
     * @param LU (Input) Compact LU storage.
     * @param L (Output) Storage of the L matrix.
     * @param U (Output) Storage of the U matrix.
     */
    compactToFull(m: number, isImaginaryPart: boolean, LU: ArrayLike<number>, L: DataBlock, U: DataBlock): void;

}

/**
 * Backend for singular value decomposition.
 */
export interface ISvdBackend {
    /**
     * Singular value decomposition for a real matrix A such that A = USV^T.
     * Adapted from Numerical Recipes.
     * @param m Number of rows in A.
     * @param n Number of columns in A.
     * @param computeUV If set to false, only singular values will be computed.
     *                  In this case, A will still be overwritten, but V will be
     *                  untouched.
     * @param reA (Input/Output) m x n Matrix A. Will be overwritten as U.
     * @param reS (Output) A n-element vector of singular values sorted in
     *            descending order.
     * @param reV (Output) n x n matrix V.
     */
    svd(m: number, n: number, computeUV: boolean, reA: DataBlock, reS: DataBlock, reV: DataBlock): void;

    /**
     * Singular value decomposition for a complex matrix A such that A = USV^H.
     * @param m Number of rows in A.
     * @param n Number of columns in A.
     * @param computeUV If set to false, only singular values will be computed.
     *                  In this case, A will still be overwritten, but V will be
     *                  untouched.
     * @param reA (Input/Output) Real part of the m x n Matrix A. Will be
     *            overwritten with the real part of U.
     * @param imA (Input/Output) Imaginary part of the m x n Matrix A. Will be
     *            overwritten with the imaginary part of U.
     * @param reS (Output) A n-element vector of singular values sorted in
     *            descending order.
     * @param reV (Output) Real part of the n x n matrix V. 
     * @param imV (Output) Imaginary part of the n x n matrix V. 
     */
    csvd(m: number, n: number, computeUV: boolean, reA: DataBlock,
        imA: DataBlock, reS: DataBlock, reV: DataBlock, imV: DataBlock): void;
}

/**
 * Backend for QR decomposition.
 */
export interface IQRBackend {
    /**
     * Performs QR decomposition with column pivoting such that AP = QR.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param a (Input/Output) Matrix A. Will be overwritten with R.
     * @param q (Output) Matrix Q. Must be initialized with zeros.
     * @param p (Output) Matrix P. Must be initialized with zeros.
     */
    qr(m: number, n: number, a: DataBlock, q: DataBlock, p: DataBlock): void;

    /**
     * Obtains the least square solution using QR decomposition such that
     * ||A X - B|| is minimized, where both A and B are real.
     * When A is rank deficient, there are infinitely many solutions. This
     * function will only return one solution satisfying the normal equation
     * (assuming free variables are all zeros).
     * @param m 
     * @param n 
     * @param p 
     * @param a (Input/Destroyed) m * n Matrix A.
     * @param b (Input/Destroyed) m x p matrix B. 
     * @param x (Output) n x p matrix X. Must be initialized to zeros.
     * @returns The estimated rank of A.
     */
    qrSolve(m: number, n: number, p: number, a: DataBlock, b: DataBlock, x: DataBlock): number;

    /**
     * Obtains the least square solution using QR decomposition such that
     * ||A X - B|| is minimized, where A is real and B is complex.
     * When A is rank deficient, there are infinitely many solutions. This
     * function will only return one solution satisfying the normal equation
     * (assuming free variables are all zeros).
     * @param m 
     * @param n 
     * @param p 
     * @param a (Input/Destroyed) m * n Matrix A.
     * @param br (Input/Destroyed) Real part of the m x p matrix B. 
     * @param bi (Input/Destroyed) Imaginary part of the m x p matrix B. 
     * @param xr (Output) Real part of the n x p matrix X. Must be initialized
     *           to zeros.
     * @param xi (Output) Imaginary part of the n x p matrix X. Must be
     *           initialized to zeros.
     * @returns The estimated rank of A.
     */
    qrSolve2(m: number, n: number, p: number, a: DataBlock, br: DataBlock, bi: DataBlock,
             xr: DataBlock, xi: DataBlock): number;

    /**
     * Performs complex QR decomposition with column pivoting such that AP = QR.
     * @param m Number of rows.
     * @param n Number of columns.
     * @param ar (Input/Output) Real part of A. Will be overwritten with the
     *                          real part of R.
     * @param ai (Input/Output) Imaginary part of A. Will be overwritten with
     *                          the imaginary part of R.
     * @param qr (Output) Real part of Q. Must be initialized with zeros.
     * @param qi (Output) Imaginary part of Q. Must be initialized with zeros.
     * @param p (Output) Matrix P. Must be initialized with zeros.
     */
    cqr(m: number, n: number, ar: DataBlock, ai: DataBlock, qr: DataBlock, qi: DataBlock, p: DataBlock): void;

    /**
     * Obtains the least square solution using QR decomposition such that
     * ||A X - B|| is minimized, where both A and B are complex.
     * When A is rank deficient, there are infinitely many solutions. This
     * function will only return one solution satisfying the normal equation
     * (assuming free variables are all zeros).
     * @param m 
     * @param n 
     * @param p 
     * @param ar (Input/Destroyed) Real part of the m x n matrix A.
     * @param ai (Input/Destroyed) Imaginary part of the m x n matrix A.
     * @param br (Input/Destroyed) Real part of the m x p matrix B. 
     * @param bi (Input/Destroyed) Imaginary part of the m x p matrix B. 
     * @param xr (Output) Real part of the n x p matrix X. Must be initialized
     *           to zeros.
     * @param xi (Output) Imaginary part of the n x p matrix X. Must be
     *           initialized to zeros.
     * @returns The estimated rank of A.
     */
    cqrSolve(m: number, n: number, p: number, ar: DataBlock, ai: DataBlock,
             br: DataBlock, bi: DataBlock, xr: DataBlock, xi: DataBlock): number;
}

/**
 * Backend for eigen-decomposition.
 */
export interface IEigenBackend {

    /**
     * Performs eigendecomposition for real symmetric matrices.
     * @param n Dimension of the matrix.
     * @param reA (Input) Matrix data.
     * @param lambda (Output) Eigenvalues.
     * @param reE (Output) Eigenvectors.
     */
    rs(n: number, reA: ArrayLike<number>, lambda: DataBlock, matz: boolean, reE: DataBlock): void;

    /**
     * Performs eigendecomposition of general real matrices.  The
     * eigenvectors are unnormalized.
     * @param n Dimension of the matrix.
     * @param a (Input/Destroyed)
     * @param wr (Output)
     * @param wi (Output)
     * @param matz Specifies whether the eigenvectors are computed. If set to
     *             false, zr and zi should be set to [].
     * @param zr (Output)
     * @param zi (Output)
     */
    rg(n: number, a: DataBlock, wr: DataBlock, wi: DataBlock, matz: boolean, zr: DataBlock, zi: DataBlock): void

    /**
     * Eigendecomposition of a Hermitian matrix.
     * @param n Dimension of the matrix.
     * @param ar (Input/Destroyed) Real part of the input matrix. Will be
     *           destroyed.
     * @param ai (Input/Destroyed) Imaginary part of the input matrix. Will be
     *           destroyed.
     * @param w (Output) Eigenvalues.
     * @param matz Specifies whether the eigenvectors are computed. If set to
     *             false, zr and zi should be set to [].
     * @param zr (Output) Real part of the eigenvectors.
     * @param zi (Output) Imaginary part of the eigenvectors.
     */
    ch(n: number, ar: DataBlock, ai: DataBlock, w: DataBlock, matz: boolean, zr: DataBlock, zi: DataBlock): void;

    /**
     * Performs eigendecomposition for general complex matrices. The
     * eigenvectors are unnormalized.
     * @param n Dimension of the matrix.
     * @param ar (Input/Destroyed)
     * @param ai (Input/Destroyed)
     * @param wr (Output)
     * @param wi (Output)
     * @param matz Specifies whether the eigenvectors are computed. If set to
     *             false, zr and zi should be set to [].
     * @param zr (Output)
     * @param zi (Output)
     */
    cg(n: number, ar: DataBlock, ai: DataBlock, wr: DataBlock, wi: DataBlock, matz: boolean,
       zr: DataBlock, zi: DataBlock): void

}

/**
 * Backend for Cholesky decomposition.
 */
export interface ICholeskyBackend {

    /**
     * In-place Cholesky decomposition for real matrices.
     * @param n Dimension of the matrix.
     * @param a (Input/Output) Matrix data stored in row major order. Only the
     *          lower triangular part will be used. If successful, the lower
     *          triangular part will be replaced with the decomposition.
     * @returns An integer p. If the decomposition is successful, p = 0. If
     *          p > 0, it implies that the subroutine detects that the input
     *          matrix is not positive definite when processing the p-th row.
     */
    chol(n: number, a: DataBlock): number;

    /**
     * In-place Cholesky decomposition for complex matrices.
     * @param n Dimension of the matrix.
     * @param reA (Input/Output) Real part of the matrix data stored in row
     *          major order. Only the lower triangular part will be used. If
     *          successful, the lower triangular part will be replaced with the
     *          decomposition.
     * @param imA (Input/Output) Imaginary part of the matrix data stored in row
     *          major order. Only the lower triangular part will be used. If
     *          successful, the lower triangular part will be replaced with the
     *          decomposition.
     * @returns An integer p. If the decomposition is successful, p = 0. If
     *          p > 0, it implies that the subroutine detects that the input
     *          matrix is not positive definite when processing the p-th row.
     */
    cchol(n: number, reA: DataBlock, imA: DataBlock): number;

}

/**
 * Backend for solving special linear systems.
 */
export interface ISpecialLinearSystemSolverBackend {

      /**
       * Solves a general real upper triangular system AX = B, where A is not
       * necessary a square matrix.
       * If m > n, the m - n extra rows will be ignored.
       * If m < n, the free variables will be set to zeros.
       * @param m
       * @param n
       * @param p
       * @param a (Input) The m x n matrix A.
       * @param b (Input) The m x p matrix B.
       * @param x (Output) The n x p matrix X.
       */
      solveGUTReal(m: number, n: number, p: number, a: ArrayLike<number>, b: ArrayLike<number>, x: DataBlock): void;

      /**
       * Solves a real upper triangular system AX = B, where A is a square
       * matrix.
       * @param m
       * @param p
       * @param a (Input) The m x m matrix A.
       * @param b (Input/Output) The m x p matrix B. Will be overwritten with X.
       */
      solveUTReal(m: number, p: number, a: ArrayLike<number>, b: DataBlock): void;
  
      /**
       * Solves a general complex upper triangular system.
       */
      solveGUTComplex(m: number, n: number, p: number, reA: ArrayLike<number>, imA: ArrayLike<number>,
                     reB: ArrayLike<number>, imB: ArrayLike<number>, reX: DataBlock, imX: DataBlock): void;

      /**
       * Solves a general real lower triangular system.
       */
      solveGLTReal(m: number, n: number, p: number, a: ArrayLike<number>, b: ArrayLike<number>, x: DataBlock): void;
      
      /**
       * Solves a general complex lower triangular system.
       */
      solveGLTComplex(m: number, n: number, p: number, reA: ArrayLike<number>, imA: ArrayLike<number>,
                     reB: ArrayLike<number>, imB: ArrayLike<number>, reX: DataBlock, imX: DataBlock): void;

}
