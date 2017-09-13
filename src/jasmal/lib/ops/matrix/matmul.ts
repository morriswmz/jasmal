import { DataBlock } from '../../commonTypes';
import { MatrixModifier } from './definition';

export interface IMatrixMultiplicationBackend {

    mulmm(m: number, n: number, p: number, modB: MatrixModifier, A: ArrayLike<number>,
          B: ArrayLike<number>, C: DataBlock): void;
    
    cmulmm(m: number, n: number, p: number, modB: MatrixModifier,
           reA: ArrayLike<number>, imA: ArrayLike<number>, reB: ArrayLike<number>,
           imB: ArrayLike<number>, reC: DataBlock, imC: DataBlock): void;

}

export class BuiltInMMB implements IMatrixMultiplicationBackend {
    
    public mulmm(m: number, n: number, p: number, modB: MatrixModifier, A: ArrayLike<number>,
                 B: ArrayLike<number>, C: DataBlock): void
    {
        if (n === 1) {
            this._mulvv(m, p, A, B, C);
        } else if (m === 1) {
            this._mulvm(n, p, modB, A, B, C);
        } else {
            this._mulmm(m, n, p, modB, A, B, C);
        }
    }

    private _mulvv(m: number, p: number, A: ArrayLike<number>, B: ArrayLike<number>, C: DataBlock): void {
        for (let i = 0;i < m;i++) {
            for (let j = 0;j < p;j++) {
                C[i * p + j] = A[i] * B[j]; 
            }
        }
    }

    private _mulvm(n: number, p: number, mobB: MatrixModifier, A: ArrayLike<number>, B: ArrayLike<number>, C: DataBlock): void {
        if (mobB === MatrixModifier.None) {
            // A*B | A: 1 x n, B: n x p, C: 1 x p
            for (let j = 0;j < p;j++) {
                C[j] = A[0] * B[j];
            }
            for (let i = 1;i < n;i++) {
                for (let j = 0;j < p;j++) {
                    C[j] += A[i] * B[i * p + j];
                }
            }
        } else if (mobB === MatrixModifier.Transposed || mobB === MatrixModifier.Hermitian) {
            // A*B^T | A: 1 x n, B^T: n x p, C: 1 x p
            for (let i = 0;i < p;i++) {
                let acc = A[0] * B[i * n];
                for (let j = 1;j < n;j++) {
                    acc += A[j] * B[i * n + j];
                }
                C[i] = acc;
            }
        }
    }

    private _mulmm(m: number, n: number, p: number, modB: MatrixModifier, A: ArrayLike<number>, B: ArrayLike<number>, C: DataBlock): void {
        let i: number, j: number, k: number, acc: number;
        if (modB === MatrixModifier.None) {
            // A*B where A: m x n, B: n x p
            if (n < 16 && p < 16) {
                // use naive implementation for small matrices
                for (j = 0;j < p;j++) {
                    for (i = 0;i < m;i++) {
                        acc = 0;
                        for (k = 0;k < n;k++) {
                            acc += A[i * n + k] * B[k * p + j];
                        }
                        C[i * p + j] = acc;
                    }
                }
            } else {
                // use cached columns for large B
                let columnCache: DataBlock = p === 1 ? B : new Array(n);
                for (j = 0;j < p;j++) {
                    // cache j-th column of B
                    if (p !== 1) {
                        for (k = 0;k < n;k++) {
                            columnCache[k] = B[k * p + j];
                        }
                    }
                    // evaluate j-th column of C
                    for (i = 0;i < m;i++) {
                        let offset = i * n;
                        let acc = A[offset] * columnCache[0];
                        k = 1;
                        // Note: further loop unrolling does not help
                        for (;k < n - 1;k += 2) {
                            acc += A[offset + k] * columnCache[k] + 
                                   A[offset + k + 1] * columnCache[k + 1];
                        }
                        if (k === n - 1) {
                            acc += A[offset + k] * columnCache[k];
                        }
                        C[i * p + j] = acc;
                    }
                }
            }
        } else if (modB === MatrixModifier.Transposed || modB === MatrixModifier.Hermitian) {
            // A*B^T where A: m x n, B^T: n x p
            // no need to cache columns here
            for (i = 0;i < m;i++) {
                for (j = 0;j < p;j++) {
                    let offsetA = i * n, offsetB = j * n;
                    let acc = A[offsetA] * B[offsetB];
                    for (k = 1;k < n;k++) {
                        acc += A[offsetA + k] * B[offsetB + k];
                    }
                    C[i * p + j] = acc;
                }
            }
        }
    }

    public cmulmm(m: number, n: number, p: number, modB: MatrixModifier,
                  reA: ArrayLike<number>, imA: ArrayLike<number>, reB: ArrayLike<number>,
                  imB: ArrayLike<number>, reC: DataBlock, imC: DataBlock): void
    {
        if (n === 1) {
            this._cmulvv(m, p, modB, reA, imA, reB, imB, reC, imC);
        } else if (m === 1) {
            this._cmulvm(n, p, modB, reA, imA, reB, imB, reC, imC);
        } else {
            this._cmulmm(m, n, p, modB, reA, imA, reB, imB, reC, imC);
        }
    }

    private _cmulvv(m: number, p: number, modB: number, reA: ArrayLike<number>, imA: ArrayLike<number>,
                    reB: ArrayLike<number>, imB: ArrayLike<number>, reC: DataBlock, imC: DataBlock): void
    {
        if (modB === MatrixModifier.None || modB === MatrixModifier.Transposed) {
            // a*b, a*b^T
            for (let i = 0;i < m;i++) {
                for (let j = 0;j < p;j++) {
                    reC[i * p + j] = reA[i] * reB[j] - imA[i] * imB[j]; 
                    imC[i * p + j] = reA[i] * imB[j] + imA[i] * reB[j]; 
                }
            }
        } else if (modB === MatrixModifier.Hermitian) {
            // a*b^H, imB -> -imB
            for (let i = 0;i < m;i++) {
                for (let j = 0;j < p;j++) {
                    reC[i * p + j] = reA[i] * reB[j] + imA[i] * imB[j]; 
                    imC[i * p + j] = - reA[i] * imB[j] + imA[i] * reB[j]; 
                }
            }
        }
    }

    private _cmulvm(n: number, p: number, modB: number, reA: ArrayLike<number>, imA: ArrayLike<number>,
                    reB: ArrayLike<number>, imB: ArrayLike<number>, reC: DataBlock, imC: DataBlock): void
    {
        let i: number, j: number;
        let accRe: number, accIm: number;
        if (modB === MatrixModifier.None) {
            // a*B where a: 1 x n, B: n x p
            for (j = 0;j < p;j++) {
                reC[j] = reA[0] * reB[j] - imA[0] * imB[j];
                imC[j] = reA[0] * imB[j] + imA[0] * reB[j];
            }
            for (i = 1;i < n;i++) {
                for (j = 0;j < p;j++) {
                    reC[j] += reA[i] * reB[i * p + j] - imA[i] * imB[i * p + j];
                    imC[j] += reA[i] * imB[i * p + j] + imA[i] * reB[i * p + j];
                }
            }
        } else if (modB === MatrixModifier.Transposed) {
            // a*B^T where a: 1 x n, B^T: n x p
            for (j = 0;j < p;j++) {
                accRe = reA[0] * reB[j * n] - imA[0] * imB[j * n];
                accIm = reA[0] * imB[j * n] + imA[0] * reB[j * n];
                for (i = 1;i < n;i++) {
                    accRe += reA[i] * reB[j * n + i] - imA[i] * imB[j * n + i];
                    accIm += reA[i] * imB[j * n + i] + imA[i] * reB[j * n + i];
                }
                reC[j] = accRe;
                imC[j] = accIm;
            }
        } else {
            // a*B^H where a: 1 x n, B^H: n x p
            for (j = 0;j < p;j++) {
                accRe = reA[0] * reB[j * n] + imA[0] * imB[j * n];
                accIm = - reA[0] * imB[j * n] + imA[0] * reB[j * n];
                for (i = 1;i < n;i++) {
                    accRe += reA[i] * reB[j * n + i] + imA[i] * imB[j * n + i];
                    accIm += - reA[i] * imB[j * n + i] + imA[i] * reB[j * n + i];
                }
                reC[j] = accRe;
                imC[j] = accIm;
            }
        }
    }

    private _cmulmm(m: number, n: number, p: number, modB: number, reA: ArrayLike<number>,
                    imA: ArrayLike<number>, reB: ArrayLike<number>, imB: ArrayLike<number>,
                    reC: DataBlock, imC: DataBlock): void
    {
        let i: number, j: number, k: number;
        let accRe: number, accIm: number;
        if (modB === 0) {
            // A*B where, A: m x n, B: n x p
            if (n < 16 && p < 16) {
                // use naive implementation for small matrices
                for (j = 0;j < p;j++) {
                    for (i = 0;i < m;i++) {
                        accRe = 0;
                        accIm = 0;
                        for (k = 0;k < n;k++) {
                            accRe += reA[i * n + k] * reB[k * p + j] - imA[i * n + k] * imB[k * p + j];
                            accIm += reA[i * n + k] * imB[k * p + j] + imA[i * n + k] * reB[k * p + j];
                        }
                        reC[i * p + j] = accRe;
                        imC[i * p + j] = accIm;
                    }
                }
            } else {
                // use column caches
                let columnCacheRe: DataBlock = p === 1 ? reB : new Array(n);
                let columnCacheIm: DataBlock = p === 1 ? imB : new Array(n);
                for (j = 0;j < p;j++) {
                    // cache j-th column of B
                    if (p !== 1) {
                        for (k = 0;k < n;k++) {
                            columnCacheRe[k] = reB[k * p + j];
                            columnCacheIm[k] = imB[k * p + j];
                        }
                    }
                    // evaluate j-th column of C
                    for (i = 0;i < m;i++) {
                        accRe = reA[i * n] * columnCacheRe[0] - imA[i * n] * columnCacheIm[0];
                        accIm = reA[i * n] * columnCacheIm[0] + imA[i * n] * columnCacheRe[0];
                        for (k = 1;k < n;k++) {
                            accRe += reA[i * n + k] * columnCacheRe[k] - imA[i * n + k] * columnCacheIm[k];
                            accIm += reA[i * n + k] * columnCacheIm[k] + imA[i * n + k] * columnCacheRe[k];
                        }
                        reC[i * p + j] = accRe;
                        imC[i * p + j] = accIm;
                    }
                }
            }
        } else if (modB === 1) {
            // A*B^T where, A: m x n, B^T: n x p
            for (i = 0;i < m;i++) {
                // evaluate j-th column of C
                for (j = 0;j < p;j++) {
                    accRe = reA[i * n] * reB[j * n] - imA[i * n] * imB[j * n];
                    accIm = reA[i * n] * imB[j * n] + imA[i * n] * reB[j * n];
                    for (k = 1;k < n;k++) {
                        accRe += reA[i * n + k] * reB[j * n + k] - imA[i * n + k] * imB[j * n + k];
                        accIm += reA[i * n + k] * imB[j * n + k] + imA[i * n + k] * reB[j * n + k];
                    }
                    reC[i * p + j] = accRe;
                    imC[i * p + j] = accIm;
                }
            }
        } else {
            // A*B^H where, A: m x n, B^H: n x p
            for (i = 0;i < m;i++) {
                // evaluate j-th column of C
                for (j = 0;j < p;j++) {
                    accRe = reA[i * n] * reB[j * n] + imA[i * n] * imB[j * n];
                    accIm = - reA[i * n] * imB[j * n] + imA[i * n] * reB[j * n];
                    for (k = 1;k < n;k++) {
                        accRe += reA[i * n + k] * reB[j * n + k] + imA[i * n + k] * imB[j * n + k];
                        accIm += - reA[i * n + k] * imB[j * n + k] + imA[i * n + k] * reB[j * n + k];
                    }
                    reC[i * p + j] = accRe;
                    imC[i * p + j] = accIm;
                }
            }
        }
    }

}
