import { IBlaoBackend } from '../backend';
import { DataBlock } from '../../commonTypes';
import { MatrixModifier } from '../modifiers';

export class BuiltInBlao implements IBlaoBackend {
    
    private _chunkSize: number;
    
    constructor(chunkSize: number = 32) {
        this._chunkSize = chunkSize;
    }

    public swap(n: number, A: DataBlock, B: DataBlock): void {
        if (n >= A.length || n >= B.length) {
            throw new Error('n cannot be larger than the number of elements in A or B.');
        }
        let tmp: number;
        for (let i = 0;i < n;i++) {
            tmp = A[i];
            A[i] = B[i];
            B[i] = tmp;
        }
    }

    public scale(alpha: number, A: DataBlock): void {
        if (alpha === 1) {
            return;
        }
        for (let i = 0;i < A.length;i++) {
            A[i] = alpha * A[i];
        }
    }

    public cscale(reAlpha: number, imAlpha: number, reA: DataBlock, imA: DataBlock): void {
        if (reA.length !== imA.length) {
            throw new Error('Real part and imaginary part must have the same number of elements.')
        }
        if (reAlpha === 1 && imAlpha === 0) {
            return;
        }
        let tmp: number;
        for (let i = 0;i < reA.length;i++) {
            tmp = reA[i];
            reA[i] = reAlpha * tmp - imAlpha * imA[i];
            imA[i] = reAlpha * imA[i] + imAlpha * tmp;
        }
    }

    public transpose(m: number, n: number, A: ArrayLike<number>, B: DataBlock): void {
        if (B === A) {
            throw new Error('In-place transpose is not supported.');
        }
        var blockSize = this._chunkSize;
        for (let ii = 0; ii < m; ii += blockSize) {
            for (let jj = 0; jj < n; jj += blockSize) {
                let iMax = ii + Math.min(blockSize, m - ii);
                let jMax = jj + Math.min(blockSize, n - jj);
                for (let i = ii; i < iMax; i++) {
                    for (let j = jj; j < jMax; j++) {
                        B[j * m + i] = A[i * n + j];
                    }
                }
            }
        }
    }

    public hermitian(m: number, n: number, reA: ArrayLike<number>, imA: ArrayLike<number>, reB: DataBlock, imB: DataBlock): void {
        let blockSize = this._chunkSize;
        // real part
        this.transpose(m, n, reA, reB);
        // imaginary part
        if (imA === imB) {
            throw new Error('In-place Hermitian is not supported.');
        }
        for (let ii = 0; ii < m; ii += blockSize) {
            for (let jj = 0; jj < n; jj += blockSize) {
                let iMax = ii + Math.min(blockSize, m - ii);
                let jMax = jj + Math.min(blockSize, n - jj);
                for (let i = ii; i < iMax; i++) {
                    for (let j = jj; j < jMax; j++) {
                        imB[j * m + i] = -imA[i * n + j];
                    }
                }
            }
        }
    }

    public ger(alpha: number, x: ArrayLike<number>, y: ArrayLike<number>, A: DataBlock): void {
        const m = x.length;
        const n = y.length;
        if (A.length !== m * n) {
            throw new Error('The number of element in A does not match that in x * y^T.');
        }
        let i: number, j: number;
        if (alpha === 0) {
            return;
        } else if (alpha === 1) {
            for (i = 0;i < m;i++) {
                for (j = 0;j < n;j++) {
                    A[i * n + j] += x[i] * y[j];
                }
            }
        } else {
            for (i = 0;i < m;i++) {
                for (j = 0;j < n;j++) {
                    A[i * n + j] += alpha * x[i] * y[j];
                }
            }
        }
    }

    private _checkCgerArgs(reX: ArrayLike<number>, imX: ArrayLike<number>,
                           reY: ArrayLike<number>, imY: ArrayLike<number>,
                           reA: ArrayLike<number>, imA: ArrayLike<number>)
    {
        if (imX.length !== reX.length) {
            throw new Error('For x, the real part and imaginary part must have the same number of elements.');
        }
        if (imY.length !== reY.length) {
            throw new Error('For y, the real part and imaginary part must have the same number of elements.');
        }
        if (reA.length !== reX.length * reY.length) {
            throw new Error('The number of element in A does not match that in x * y^T.');
        }
        if (reA.length !== imA.length) {
            throw new Error('For A, the real part and imaginary part must have the same number of elements.');
        }
    }

    public cgeru(reAlpha: number, imAlpha: number, reX: ArrayLike<number>, imX: ArrayLike<number>,
        reY: ArrayLike<number>, imY: ArrayLike<number>, reA: DataBlock, imA: DataBlock): void
    {
        this._checkCgerArgs(reX, imX, reY, imY, reA, imA);
        const m = reX.length;
        const n = reY.length;
        let i: number, j: number;
        if (imAlpha === 0) {
            if (reAlpha === 0) {
                return;
            } else if (reAlpha === 1) {
                for (i = 0;i < m;i++) {
                    for (j = 0;j < n;j++) {
                        reA[i * n + j] += reX[i] * reY[j] - imX[i] * imY[j]; 
                        imA[i * n + j] += imX[i] * reY[j] + reX[i] * imY[j]; 
                    }
                }
                return;
            }
        }
        let re: number, im: number;
        for (i = 0;i < m;i++) {
            for (j = 0;j < n;j++) {
                re = reX[i] * reY[j] - imX[i] * imY[j];
                im = imX[i] * reY[j] + reX[i] * imY[j];
                reA[i * n + j] += re * reAlpha - im * imAlpha; 
                imA[i * n + j] += re * imAlpha + im * reAlpha; 
            }
        }
    }

    public cgerc(reAlpha: number, imAlpha: number, reX: ArrayLike<number>, imX: ArrayLike<number>,
                 reY: ArrayLike<number>, imY: ArrayLike<number>, reA: DataBlock, imA: DataBlock): void
    {
        this._checkCgerArgs(reX, imX, reY, imY, reA, imA);
        const m = reX.length;
        const n = reY.length;
        let i: number, j: number;
        if (imAlpha === 0) {
            if (reAlpha === 0) {
                return;
            } else if (reAlpha === 1) {
                for (i = 0;i < m;i++) {
                    for (j = 0;j < n;j++) {
                        reA[i * n + j] += reX[i] * reY[j] + imX[i] * imY[j]; 
                        imA[i * n + j] += imX[i] * reY[j] - reX[i] * imY[j]; 
                    }
                }
                return;
            }
        }
        let re: number, im: number;
        for (i = 0;i < m;i++) {
            for (j = 0;j < n;j++) {
                re = reX[i] * reY[j] + imX[i] * imY[j];
                im = imX[i] * reY[j] - reX[i] * imY[j];
                reA[i * n + j] += re * reAlpha - im * imAlpha; 
                imA[i * n + j] += re * imAlpha + im * reAlpha; 
            }
        }
    }

    public gemv(m: number, n: number, alpha: number, A: ArrayLike<number>, modA: MatrixModifier,
                 x: ArrayLike<number>, beta: number, y: DataBlock): void
    {
        if (x.length !== n) {
            throw new Error('The length of x must be equal to n.');
        }
        if (y.length !== m) {
            throw new Error('The length of y must be equal to m.');
        }
        if (A.length !== m * n) {
            throw new Error('The number of elements in A must be equal to m * n.');
        }

        if (alpha === 0) {
            return;
        }
        if (beta !== 1) {
            this.scale(beta, y);
        }
        let i: number, j: number;
        let acc: number;
        if (modA === MatrixModifier.None) {
            for (i = 0;i < m;i++) {
                acc = 0;
                for (j = 0;j < n;j++) {
                    acc += A[i * n + j] * x[j];
                }
                y[i] += alpha === 1 ? acc : alpha * acc;
            }
        } else {
            for (i = 0;i < m;i++) {
                acc = 0;
                for (j = 0;j < n;j++) {
                    acc += A[j * m + i] * x[j];
                }
                y[i] += alpha === 1 ? acc : alpha * acc;
            }
        }
    }

    public cgemv(m: number, n: number, reAlpha: number, imAlpha: number,
                 reA: ArrayLike<number>, imA: ArrayLike<number>, modA: MatrixModifier,
                 reX: ArrayLike<number>, imX: ArrayLike<number>, reBeta: number, imBeta: number,
                 reY: DataBlock, imY: DataBlock): void
    {
        if (reX.length !== n) {
            throw new Error('The length of x must be equal to n.');
        }
        if (reX.length !== imX.length) {
            throw new Error('The real part and the imaginary part of x must have the same number of elements.');
        }
        if (reY.length !== m) {
            throw new Error('The length of y must be equal to m.');
        }
        if (reY.length !== imY.length) {
            throw new Error('The real part and the imaginary part of y must have the same number of elements.');
        }
        if (reA.length !== m * n) {
            throw new Error('The number of elements in A must be equal to m * n.');
        }
        if (reA.length !== imA.length) {
            throw new Error('The real part and the imaginary part of A must have the same number of elements.');
        }
        
        if (reAlpha === 0 && imAlpha === 0) {
            return;
        }
        if (reBeta !== 1 || imBeta !== 0) {
            this.cscale(reBeta, imBeta, reY, imY);
        }
        const isAlphaOne = reAlpha === 1 && imAlpha === 0;
        let i: number, j: number;
        let accRe: number, accIm: number;
        if (modA === MatrixModifier.None) {
            for (i = 0;i < m;i++) {
                accRe = 0;
                accIm = 0;
                for (j = 0;j < n;j++) {
                    accRe += reA[i * n + j] * reX[j] - imA[i * n + j] * imX[j];
                    accIm += reA[i * n + j] * imX[j] + imA[i * n + j] * reX[j];
                }
                if (isAlphaOne) {
                    reY[i] += accRe;
                    imY[i] += accIm;
                } else {
                    reY[i] += accRe * reAlpha - accIm * imAlpha;
                    imY[i] += accRe * imAlpha + accIm * reAlpha;
                }
            }
        } else {
            for (i = 0;i < m;i++) {
                accRe = 0;
                accIm = 0;
                for (j = 0;j < n;j++) {
                    accRe += reA[j * m + i] * reX[j] - imA[j * m + i] * imX[j];
                    accIm += reA[j * m + i] * imX[j] + imA[j * m + i] * reX[j];
                }
                if (isAlphaOne) {
                    reY[i] += accRe;
                    imY[i] += accIm;
                } else {
                    reY[i] += accRe * reAlpha - accIm * imAlpha;
                    imY[i] += accRe * imAlpha + accIm * reAlpha;
                }
            }
        }
    }

    public gemm(m: number, n: number, k: number, alpha: number, A: ArrayLike<number>,
                 B: ArrayLike<number>, modB: MatrixModifier, beta: number, C: DataBlock): void
    {
        if (k === 1) {
            // row vector - column vector
            if (beta !== 1) {
                this.scale(beta, C);
            }
            this.ger(alpha, A, B, C);
        } else if (m === 1) {
            // row vector - matrix
            // a M(B) = (M(B)^T a^T)^T
            this.gemv(n, k, alpha, B, modB === MatrixModifier.None ? MatrixModifier.Transposed : MatrixModifier.None, A, beta, C);
        } else if (n === 1) {
            // matrix - column vector
            this.gemv(m, k, alpha, A, MatrixModifier.None, B, beta, C);
        } else {
            // matrix - matrix
            this._dgemm(m, n, k, alpha, A, B, modB, beta, C);
        }
    }

    private _dgemm(m: number, n: number, k: number, alpha: number, A: ArrayLike<number>,
                   B: ArrayLike<number>, modB: MatrixModifier, beta: number, C: DataBlock): void
    {
        if (A.length !== m * k) {
            throw new Error('The number of elements in A must be m * k.');
        }
        if (B.length !== k * n) {
            throw new Error('The number of elements in B must be k * n.');
        }
        if (C.length !== m * n) {
            throw new Error('The number of elements in C must be m * n.');            
        }

        if (alpha === 0) {
            return;
        }
        if (beta !== 1) {
            this.scale(beta, C);
        }
        // TODO: is it necessary to handle the special case when alpha === 1?
        let i: number, j: number, l: number, acc: number;
        let offsetA: number, offsetB: number;
        if (modB === MatrixModifier.None) {
            // A*B where A: m x k, B: k x n
            if (m < 3 || (k < 8 && n < 8)) {
                // use naive implementation for small matrices
                for (j = 0;j < n;j++) {
                    for (i = 0;i < m;i++) {
                        acc = 0;
                        for (l = 0;l < k;l++) {
                            acc += A[i * k + l] * B[l * n + j];
                        }
                        C[i * n + j] += alpha * acc;
                    }
                }
            } else {
                // use cached columns for large B
                // TODO: should we use a typed array here?
                let columnCache: DataBlock = new Array(k);
                for (j = 0;j < n;j++) {
                    // cache j-th column of B
                    for (l = 0;l < k;l++) {
                        columnCache[l] = B[l * n + j];
                    }
                    // evaluate j-th column of C
                    offsetA = 0;
                    for (i = 0;i < m;i++) {
                        acc = A[offsetA] * columnCache[0];
                        l = 1;
                        // Note: further loop unrolling does not help
                        for (;l < k - 1;l += 2) {
                            acc += A[offsetA + l] * columnCache[l] + 
                                   A[offsetA + l + 1] * columnCache[l + 1];
                        }
                        if (l === k - 1) {
                            acc += A[offsetA + l] * columnCache[l];
                        }
                        C[i * n + j] += alpha * acc;
                        offsetA += k;
                    }
                }
            }
        } else if (modB === MatrixModifier.Transposed || modB === MatrixModifier.Hermitian) {
            // A*B^T where A: m x k, B^T: k x n
            // no need to cache columns here
            offsetA = 0;
            for (i = 0;i < m;i++) {
                offsetB = 0;
                for (j = 0;j < n;j++) {
                    acc = A[offsetA] * B[offsetB];
                    for (l = 1;l < k;l++) {
                        acc += A[offsetA + l] * B[offsetB + l];
                    }
                    C[i * n + j] += alpha * acc;
                    offsetB += k;
                }
                offsetA += k;
            }
        }
    }

    public cgemm(m: number, n: number, k: number, reAlpha: number, imAlpha: number,
                 reA: ArrayLike<number>, imA: ArrayLike<number>,
                 reB: ArrayLike<number>, imB: ArrayLike<number>, modB: MatrixModifier,
                 reBeta: number, imBeta: number, reC: DataBlock, imC: DataBlock): void
    {
        if (k === 1) {
            // row vector - column vector
            if (reBeta !== 1 || imBeta !== 0) {
                this.cscale(reAlpha, imAlpha, reC, imC);
            }
            if (modB === MatrixModifier.Hermitian) {
                this.cgerc(reAlpha, imAlpha, reA, imA, reB, imB, reC, imC);
            } else {
                this.cgeru(reAlpha, imAlpha, reA, imA, reB, imB, reC, imC);
            }
        } else if (m === 1) {
            // row vector - matrix
            this._cgemm(1, n, k, reAlpha, imAlpha, reA, imA, reB, imB, modB, reBeta, imBeta, reC, imC);
        } else if (n === 1) {
            // matrix - column vector
            this.cgemv(m, k, reAlpha, imAlpha, reA, imA, MatrixModifier.None,
                       reB, imB, reBeta, imBeta, reC, imC);
        } else {
            // matrix - matrix
            this._cgemm(m, n, k, reAlpha, imAlpha, reA, imA, reB, imB, modB, reBeta, imBeta, reC, imC);
        }
    }

    private _cgemm(m: number, n: number, k: number, reAlpha: number, imAlpha: number,
                   reA: ArrayLike<number>, imA: ArrayLike<number>,
                   reB: ArrayLike<number>, imB: ArrayLike<number>, modB: MatrixModifier,
                   reBeta: number, imBeta: number, reC: DataBlock, imC: DataBlock): void
    {
        if (reA.length !== m * k) {
            throw new Error('The number of elements in A must be m * k.');
        }
        if (reA.length !== imA.length) {
            throw new Error('The real part and the imaginary part of A must have the same number of elements.');
        }
        if (reB.length !== k * n) {
            throw new Error('The number of elements in B must be k * n.');
        }
        if (reB.length !== imB.length) {
            throw new Error('The real part and the imaginary part of A must have the same number of elements.');
        }
        if (reC.length !== m * n) {
            throw new Error('The number of elements in C must be m * n.');            
        }
        if (reC.length !== imC.length) {
            throw new Error('The real part and the imaginary part of A must have the same number of elements.');
        }

        if (reAlpha === 0 && imAlpha === 0) {
            return;
        }
        if (reBeta !== 1 || imBeta !== 0) {
            this.cscale(reBeta, imBeta, reC, imC);
        }
        // TODO: is it necessary to handle the special case when alpha === 1?
        // TODO: should we use a typed array for column caches?
        let i: number, j: number, l: number;
        let accRe: number, accIm: number;
        let offsetA: number, offsetB: number;
        if (modB === MatrixModifier.None) {
            // A*B where A: m x k, B: k x n
            if (m < 3 || (k < 8 && n < 8)) {
                // use naive implementation for small matrices
                for (j = 0;j < n;j++) {
                    // evaluate j-th column of C
                    offsetA = 0;
                    for (i = 0;i < m;i++) {
                        accRe = 0;
                        accIm = 0;
                        for (l = 0;l < k;l++) {
                            accRe += reA[offsetA + l] * reB[l * n + j] - imA[offsetA + l] * imB[l * n + j];
                            accIm += reA[offsetA + l] * imB[l * n + j] + imA[offsetA + l] * reB[l * n + j];
                        }
                        reC[i * n + j] += reAlpha * accRe - imAlpha * accIm;
                        imC[i * n + j] += reAlpha * accIm + imAlpha * accRe;
                        offsetA += k;
                    }
                }
            } else {
                let columnCacheRe: DataBlock = new Array(k);
                let columnCacheIm: DataBlock = new Array(k);
                for (j = 0;j < n;j++) {
                    // cache j-th column of B
                    for (l = 0;l < k;l++) {
                        columnCacheRe[l] = reB[l * n + j];
                        columnCacheIm[l] = imB[l * n + j];
                    }
                    // evaluate j-th column of C
                    offsetA = 0;
                    for (i = 0;i < m;i++) {
                        accRe = 0;
                        accIm = 0;
                        for (l = 0;l < k;l++) {
                            accRe += reA[offsetA + l] * columnCacheRe[l] - imA[offsetA + l] * columnCacheIm[l];
                            accIm += reA[offsetA + l] * columnCacheIm[l] + imA[offsetA + l] * columnCacheRe[l];
                        }
                        reC[i * n + j] += reAlpha * accRe - imAlpha * accIm;
                        imC[i * n + j] += reAlpha * accIm + imAlpha * accRe;
                        offsetA += k;
                    }
                }
            }
        } else if (modB === MatrixModifier.Transposed) {
            // A*B^T where A: m x k, B^T: k x n
            // no need to cache columns here
            offsetA = 0;
            for (i = 0;i < m;i++) {
                offsetB = 0;
                for (j = 0;j < n;j++) {
                    accRe = 0;
                    accIm = 0;
                    for (l = 0;l < k;l++) {
                        accRe += reA[offsetA + l] * reB[offsetB + l] - imA[offsetA + l] * imB[offsetB + l];
                        accIm += reA[offsetA + l] * imB[offsetB + l] + imA[offsetA + l] * reB[offsetB + l];
                    }
                    reC[i * n + j] += reAlpha * accRe - imAlpha * accIm;
                    imC[i * n + j] += reAlpha * accIm + imAlpha * accRe;
                    offsetB += k;
                }
                offsetA += k;
            }
        } else {
            // A*B^H where A: m x k, B^H: k x n
            // no need to cache columns here
            offsetA = 0;
            for (i = 0;i < m;i++) {
                offsetB = 0;
                for (j = 0;j < n;j++) {
                    accRe = 0;
                    accIm = 0;
                    for (l = 0;l < k;l++) {
                        accRe += reA[offsetA + l] * reB[offsetB + l] + imA[offsetA + l] * imB[offsetB + l];
                        accIm += imA[offsetA + l] * reB[offsetB + l] - reA[offsetA + l] * imB[offsetB + l];
                    }
                    reC[i * n + j] += reAlpha * accRe - imAlpha * accIm;
                    imC[i * n + j] += reAlpha * accIm + imAlpha * accRe;
                    offsetB += k;
                }
                offsetA += k;
            }
        }
    }

}
