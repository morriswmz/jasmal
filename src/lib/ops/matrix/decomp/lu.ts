import { Tensor } from '../../../tensor';
import { CMath } from '../../../complexNumber';
import { DataBlock } from '../../../commonTypes';
import { OutputDTypeResolver } from '../../../dtype';
import { DataHelper } from "../../../helper/dataHelper";

/**
 * LUP decomposition.
 */
export class LU {
    
    /**
     * In-place LUP decomposition using Crout's algorithm.
     * @param m Dimension of the matrix.
     * @param reX (Input/Output) Real part. Will be overwritten with the LU
     *            decomposition.
     * @param p (Output) Array storing the permutation vector. For instance, if
     *          p = [2, 0, 1], then the 1st row now stores the original 3rd row.
     * @returns Induced sign of the permutation (det(P)).
     */
    public static lu(m: number, reX: DataBlock, p: DataBlock): number {
        LU._fillIndexVector(p);    
        let scales = new Array<number>(m);
        let sign = 1;
        // records scaling factor for each row
        for (let i = 0; i < m; i++) {
            let maxAbs = 0.0;
            for (let j = 0; j < m; j++) {
                let curAbs = Math.abs(reX[i * m + j]);
                if (curAbs > maxAbs) {
                    maxAbs = curAbs;
                }
            }
            scales[i] = maxAbs > 0 ? 1.0 / maxAbs : 1.0;
        }
        // performs actual LUP decomposition
        let acc: number,
            tmp: number;
        for (let j = 0; j < m; j++) {
            // solves for j-th column of U
            for (let i = 0; i < j; i++) {
                acc = reX[i * m + j];
                for (let k = 0; k < i; k++) {
                    acc -= reX[i * m + k] * reX[k * m + j];
                }
                reX[i * m + j] = acc;
            }
            // solves for i-th column of L and pivoting
            let maxAbs = 0.0,
                idxMax = -1;
            for (let i = j; i < m; i++) {
                acc = reX[i * m + j];
                for (let k = 0; k < j; k++) {
                    acc -= reX[i * m + k] * reX[k * m + j];
                }
                reX[i * m + j] = acc;
                // check abs
                let curAbs = Math.abs(acc) * scales[i];
                if (curAbs >= maxAbs) {
                    maxAbs = curAbs;
                    idxMax = i;
                }
            }
            if (idxMax !== j) {
                // we need to swap rows
                for (let k = 0; k < m; k++) {
                    tmp = reX[idxMax * m + k];
                    reX[idxMax * m + k] = reX[j * m + k];
                    reX[j * m + k] = tmp;
                }
                // flips the sign
                sign = -sign;
                // we do not need to set scales[j] because we will never use it
                // in the following iterations
                scales[idxMax] = scales[j]; 
                // record the swap
                tmp = p[j];
                p[j] = p[idxMax];
                p[idxMax] = tmp;
            }
            
            // divide by the pivot element
            if (j !== m - 1) {
                let c = reX[j * m + j];
                // if pivoting element is zero, the remaining elements are
                // also zero -> no need to perform the division
                if (c !== 0) {
                    c = 1.0 / c;
                    for (let i = j + 1; i < m; i++) {
                        reX[i * m + j] *= c;
                    }
                }
            }
        }
        return sign;
    }

    /**
     * In-place complex LUP decomposition using Crout's algorithm.
     * @param m Dimension of the matrix.
     * @param reX (Input/Output) Real part. Will be overwritten with the real
     *            part of the LU decomposition.
     * @param imX (Input/Output) Imaginary part. Will be overwritten with the
     *            imaginary part of the LU decomposition.
     * @param p (Output) Array storing the permutation vector. For instance, if
     *          p = [2, 0, 1], then the 1st row now stores the original 3rd row.
     * @returns Induced sign of the permutation (det(P)).
     */
    public static clu(m: number, reX: DataBlock, imX: DataBlock, p: DataBlock): number {
        LU._fillIndexVector(p);
        let scales = new Array<number>(m);
        let sign = 1;
        // records scaling factor for each row
        for (let i = 0; i < m; i++) {
            let maxAbs = 0.0;
            for (let j = 0; j < m; j++) {
                let curAbs = CMath.length2(reX[i * m + j], imX[i * m + j]);
                if (curAbs > maxAbs) {
                    maxAbs = curAbs;
                }
            }
            scales[i] = maxAbs > 0 ? 1.0 / maxAbs : 1.0;
        }
        // performs actual LUP decomposition
        let accRe: number, accIm: number, tmpRe: number, tmpIm: number;
        for (let j = 0; j < m; j++) {
            // solves for j-th column of U
            for (let i = 0; i < j; i++) {
                accRe = reX[i * m + j];
                accIm = imX[i * m + j];
                for (let k = 0; k < i; k++) {
                    accRe -= reX[i * m + k] * reX[k * m + j] - imX[i * m + k] * imX[k * m + j];
                    accIm -= reX[i * m + k] * imX[k * m + j] + imX[i * m + k] * reX[k * m + j];
                }
                reX[i * m + j] = accRe;
                imX[i * m + j] = accIm;
            }
            // solves for i-th column of L and pivoting
            let maxAbs = 0.0,
                idxMax = -1;
            for (let i = j; i < m; i++) {
                accRe = reX[i * m + j];
                accIm = imX[i * m + j];
                for (let k = 0; k < j; k++) {
                    accRe -= reX[i * m + k] * reX[k * m + j] - imX[i * m + k] * imX[k * m + j];
                    accIm -= reX[i * m + k] * imX[k * m + j] + imX[i * m + k] * reX[k * m + j];
                }
                reX[i * m + j] = accRe;
                imX[i * m + j] = accIm;
                // check abs
                let curAbs = CMath.length2(accRe, accIm) * scales[i];
                if (curAbs >= maxAbs) {
                    maxAbs = curAbs;
                    idxMax = i;
                }
            }
            if (idxMax !== j) {
                // we need to swap rows
                for (let k = 0; k < m; k++) {
                    tmpRe = reX[idxMax * m + k];
                    tmpIm = imX[idxMax * m + k];
                    reX[idxMax * m + k] = reX[j * m + k];
                    imX[idxMax * m + k] = imX[j * m + k];
                    reX[j * m + k] = tmpRe;
                    imX[j * m + k] = tmpIm;
                }
                // flips the sign
                sign = -sign;
                // we do not need to set scales[j] because we will never use it
                // in the following iterations
                scales[idxMax] = scales[j]; 
                // record the swap
                tmpRe = p[j];
                p[j] = p[idxMax];
                p[idxMax] = tmpRe;
            }

            // divide by the pivot element
            if (j !== m - 1) {
                let reC = reX[j * m + j];
                let imC = imX[j * m + j];
                // if pivoting element is zero, the remaining elements are
                // also zero -> no need to perform the division
                if (reC !== 0 || imC !== 0) {
                    // 1.0 / pivot element
                    [reC, imC] = CMath.cReciprocal(reC, imC);
                    for (let i = j + 1; i < m; i++) {
                        let tmp = reX[i * m + j];
                        reX[i * m + j] = tmp * reC - imX[i * m + j] * imC;
                        imX[i * m + j] = tmp * imC + imX[i * m + j] * reC;
                    }
                }
            }
        }
        return sign;
    }

    /**
     * Solves P*L*U*X = B, where P, L, U are m x m, and B is m x n.
     * B will be overwritten with the solution X.
     * @param m Number of rows in B.
     * @param n Number of columns in B.
     * @param reLU (Input) Compact storage of L and U of the PLU decomposition.
     * @param p (Input) Permutation vector.
     * @param reB (Output) Matrix B.
     */
    public static luSolve(m: number, n: number, reLU: ArrayLike<number>,
                          p: ArrayLike<number>, reB: DataBlock): void {
        if (n === 1) {
            LU._luSolveColumn(n, reLU, reB);
        } else {
            let columnCache: number[] = new Array(m);
            for (let j = 0;j < n;j++) {
                for (let i = 0;i < m;i++) {
                    columnCache[i] = reB[p[i] * n + j];
                }
                LU._luSolveColumn(m, reLU, columnCache);
                for (let i = 0;i < m;i++) {
                    reB[i * n + j] = columnCache[i];
                }
            }
        }
    }

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
    public static cluSolve(m: number, n: number, reLU: ArrayLike<number>,
                           imLU: ArrayLike<number>, p: ArrayLike<number>,
                           reB: DataBlock, imB: DataBlock): void {
        if (n === 1) {
            LU._cluSolveColumn(n, reLU, imLU, reB, imB);
        } else {
            let columnCacheRe: number[] = new Array(m);
            let columnCacheIm: number[] = new Array(m);
            for (let j = 0;j < n;j++) {
                for (let i = 0;i < m;i++) {
                    columnCacheRe[i] = reB[p[i] * n + j];
                    columnCacheIm[i] = imB[p[i] * n + j];
                }
                LU._cluSolveColumn(m, reLU, imLU, columnCacheRe, columnCacheIm);
                for (let i = 0;i < m;i++) {
                    reB[i * n + j] = columnCacheRe[i];
                    imB[i * n + j] = columnCacheIm[i];
                }
            }
        }
    }

    /**
     * Solves LUx = b in-place (b will be overridden).
     * @param n Length of b.
     * @param reLU (Input) Compact storage of LU decomposition. Will be
     *             overwritten with the solution.
     * @param reB (Output) Vector b.
     */
    public static _luSolveColumn(n: number, reLU: ArrayLike<number>, reB: DataBlock): void {
        let idxNz = -1;
        let acc: number, idxPermuted: number;
        // L^-1 P^T b
        for (let i = 0; i < n; i++) {
            acc = reB[i];
            if (idxNz >= 0) {
                // only start from the first non-zero element
                // to avoid unnecessary accumulations
                for (let k = idxNz; k < i; k++) {
                    acc -= reLU[i * n + k] * reB[k];
                }
            } else if (acc !== 0) {
                // found a non-zero element, record it
                idxNz = i;
            }
            reB[i] = acc; // stores the result
        }
        // U^-1
        for (let i = n - 1; i >= 0; i--) {
            acc = reB[i];
            for (let j = i + 1; j < n; j++) {
                acc -= reLU[i * n + j] * reB[j];
            }
            reB[i] = acc / reLU[i * n + i];
        }
    }

    /**
     * Solves LUx = b in-place (b will be overridden) for the complex case.
     * @param n Length of b.
     * @param reLU (Input) Real part of the compact storage of LU decomposition.
     * @param imLU (Input) Imaginary part of the compact storage of LU
     *             decomposition.
     * @param reB (Output) Real part of the vector b. Will be overwritten with
     *            the solution.
     * @param imB (Output) Imaginary part of the vector b. Will be overwritten
     *            with the solution.
     */
    public static _cluSolveColumn(n: number, reLU: ArrayLike<number>, imLU: ArrayLike<number>,
                                  reB: DataBlock, imB: DataBlock): void {
        let idxNz = -1;
        let accRe: number, accIm: number, idxPermuted: number;
        // L^-1 P^T b
        for (let i = 0; i < n; i++) {
            accRe = reB[i];
            accIm = imB[i];
            if (idxNz >= 0) {
                // only start from the first non-zero element
                // to avoid unnecessary accumulations
                for (let k = idxNz; k < i; k++) {
                    accRe -= reLU[i * n + k] * reB[k] - imLU[i * n + k] * imB[k];
                    accIm -= reLU[i * n + k] * imB[k] + imLU[i * n + k] * reB[k];
                }
            } else if (accRe !== 0 || accIm !== 0) {
                // found a non-zero element, record it
                idxNz = i;
            }
            // stores the result
            reB[i] = accRe;
            imB[i] = accIm;
        }
        // U^-1
        for (let i = n - 1; i >= 0; i--) {
            accRe = reB[i];
            accIm = imB[i];
            for (let j = i + 1; j < n; j++) {
                accRe -= reLU[i * n + j] * reB[j] - imLU[i * n + j] * imB[j];
                accIm -= reLU[i * n + j] * imB[j] + imLU[i * n + j] * reB[j];
            }
            [reB[i], imB[i]] = CMath.cdivCC(accRe, accIm, reLU[i * n + i], imLU[i * n + i]);
        }
    }

    /**
     * Converts a compact LU matrix to full L and U matrices.
     * @param m Dimension of the matrix.
     * @param isImaginaryPart Whether we are converting the imaginary part.
     * @param LU (Input) Compact LU storage.
     * @param L (Output) Storage of the L matrix.
     * @param U (Output) Storage of the U matrix.
     */
    public static compactToFull(m: number, isImaginaryPart: boolean, LU: ArrayLike<number>, L: DataBlock, U: DataBlock): void {
        if (!isImaginaryPart) {
            for (let i = 0; i < m; i++) {
                L[i * m + i] = 1;
            }
        }
        for (let i = 0;i < m;i++) {
            let j = 0;
            for (;j < i;j++) {
                L[i * m + j] = LU[i * m + j];
            }
            for (;j < m;j++) {
                U[i * m + j] = LU[i * m + j];
            }
        }
    }

    /**
     * Converts the permutation vector to the corresponding permutation matrix.
     * @param p (Input) Permutation vector.
     * @param P (Output) Storage for the permutation matrix P.
     */
    public static permutationToFull(p: ArrayLike<number>, P: DataBlock): void {
        for (let i = 0; i < p.length; i++) {
            P[p[i] * p.length + i] = 1;
        }
    }

    public static _fillIndexVector(p: DataBlock): void {
        for (let i = 0;i < p.length; i++) {
            p[i] = i;
        }
    }
}