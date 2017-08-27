import { DataBlock } from '../../commonTypes';

export interface IMatrixBasicSubroutines {

    mmul(dims: [number, number, number], opB: number, A: DataBlock, B: DataBlock, C: DataBlock): void;

    cmmul(dims: [number, number, number], opB: number, reA: DataBlock, imA: DataBlock, reB: DataBlock,
          imB: DataBlock, reC: DataBlock, imC: DataBlock): void;
        
    transpose(dims: [number, number], A: DataBlock, B: DataBlock): void;

    hermitian(dims: [number, number], reA: DataBlock, imA: DataBlock,
              reB: DataBlock, imB: DataBlock): void;

}

export class BuiltInMBS implements IMatrixBasicSubroutines {

    private _chunkSize: number;

    constructor(chunkSize: number = 32) {
        this._chunkSize = chunkSize;
    }

    public mmul(dims: [number, number, number], opB: number, A: DataBlock, B: DataBlock, C: DataBlock): void {
        if (opB === 2) {
            // Hermitian is equivalent to transpose for real matrices.
            opB = 1;
        }
        if (dims[1] === 1) {
            this._mmul_vv(dims, A, B, C);
        } else if (dims[0] === 1) {
            this._mmul_vm(dims, opB, A, B, C);
        } else {
            this._mmul_mm(dims, opB, A, B, C);
        }
    }

    private _mmul_vv(dims: [number, number, number], A: DataBlock, B: DataBlock, C: DataBlock): void {
        let m = dims[0], p = dims[2];
        for (let i = 0;i < m;i++) {
            for (let j = 0;j < p;j++) {
                C[i * p + j] = A[i] * B[j]; 
            }
        }
    }

    private _mmul_vm(dims: [number, number, number], opB: number, A: DataBlock, B: DataBlock, C: DataBlock): void {
        let n = dims[1], p = dims[2];
        if (opB === 0) {
            // B: n x p
            for (let j = 0;j < p;j++) {
                C[j] = A[0] * B[j];
            }
            for (let i = 1;i < n;i++) {
                for (let j = 0;j < p;j++) {
                    C[j] += A[i] * B[i * p + j];
                }
            }
        } else if (opB === 1) {
            // B: p x n
            for (let i = 0;i < p;i++) {
                let acc = A[0] * B[i * n];
                for (let j = 1;j < n;j++) {
                    acc += A[j] * B[i * n + j];
                }
                C[i] = acc;
            }
        }
    }

    private _mmul_mm(dims: [number, number, number], opB: number, A: DataBlock, B: DataBlock, C: DataBlock): void {
        let [m, n, p] = dims;
        if (opB === 0) {
            // A*B where A: m x n, B: n x p
            let columnCache: DataBlock = p === 1 ? B : new Array(n);
            for (let j = 0;j < p;j++) {
                // cache j-th column of B
                if (p !== 1) {
                    for (let k = 0;k < n;k++) {
                        columnCache[k] = B[k * p + j];
                    }
                }
                // evaluate j-th column of C
                for (let i = 0;i < m;i++) {
                    let offset = i * n;
                    let acc = A[offset] * columnCache[0];
                    let k = 1;
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
        } else if (opB === 1) {
            // A*B^T where A: m x n, B: p x n
            // no need to cache columns here
            for (let i = 0;i < m;i++) {
                for (let j = 0;j < p;j++) {
                    let offsetA = i * n, offsetB = j * n;
                    let acc = A[offsetA] * B[offsetB];
                    for (let k = 1;k < n;k++) {
                        acc += A[offsetA + k] * B[offsetB + k];
                    }
                    C[i * p + j] = acc;
                }
            }
        }
    }

    public cmmul(dims: [number, number, number], opB: number, reA: DataBlock, imA: DataBlock,
                 reB: DataBlock, imB: DataBlock, reC: DataBlock, imC: DataBlock): void {
        if (opB !== 0 && opB !== 1 && opB !== 2) {
            throw new Error(`Unsupported operation id ${opB} over matrix B.`);
        }
        if (dims[1] === 1) {
            this._cmmul_vv(dims, opB, reA, imA, reB, imB, reC, imC);
        } else if (dims[0] === 1) {
            this._cmmul_vm(dims, opB, reA, imA, reB, imB, reC, imC);
        } else {
            this._cmmul_mm(dims, opB, reA, imA, reB, imB, reC, imC);
        }
    }

    private _cmmul_vv(dims: [number, number, number], opB: number, reA: DataBlock, imA: DataBlock,
                      reB: DataBlock, imB: DataBlock, reC: DataBlock, imC: DataBlock): void {
        let m = dims[0], p = dims[2];
        if (opB === 0) {
            // a*b, a*b^T
            for (let i = 0;i < m;i++) {
                for (let j = 0;j < p;j++) {
                    reC[i * p + j] = reA[i] * reB[j] - imA[i] * imB[j]; 
                    imC[i * p + j] = reA[i] * imB[j] + imA[i] * reB[j]; 
                }
            }
        } else if (opB === 2) {
            // a*b^H, imB -> -imB
            for (let i = 0;i < m;i++) {
                for (let j = 0;j < p;j++) {
                    reC[i * p + j] = reA[i] * reB[j] + imA[i] * imB[j]; 
                    imC[i * p + j] = - reA[i] * imB[j] + imA[i] * reB[j]; 
                }
            }
        }
    }

    private _cmmul_vm(dims: [number, number, number], opB: number, reA: DataBlock, imA: DataBlock,
                      reB: DataBlock, imB: DataBlock, reC: DataBlock, imC: DataBlock): void {
        let n = dims[1], p = dims[2];
        if (opB === 0) {
            // a*B where a: 1 x n, B: n x p
            for (let j = 0;j < p;j++) {
                reC[j] = reA[0] * reB[j] - imA[0] * imB[j];
                imC[j] = reA[0] * imB[j] + imA[0] * reB[j];
            }
            for (let i = 1;i < n;i++) {
                for (let j = 0;j < p;j++) {
                    reC[j] += reA[i] * reB[i * p + j] - imA[i] * imB[i * p + j];
                    reC[j] += reA[i] * imB[i * p + j] + imA[i] * reB[i * p + j];
                }
            }
        } else if (opB === 1) {
            // a*B^T where a: 1 x n, B: p x n
            for (let j = 0;j < p;j++) {
                let accRe = reA[0] * reB[j * n] - imA[0] * imB[j * n];
                let accIm = reA[0] * imB[j * n] + imA[0] * reB[j * n];
                for (let i = 1;i < n;i++) {
                    accRe += reA[i] * reB[j * n + i] - imA[0] * imB[j * n + i];
                    accIm += reA[i] * imB[j * n + i] + imA[0] * reB[j * n + i];
                }
            }
        } else {
            // a*B^H where a: 1 x n, B: p x n
            for (let j = 0;j < p;j++) {
                let accRe = reA[0] * reB[j * n] + imA[0] * imB[j * n];
                let accIm = - reA[0] * imB[j * n] + imA[0] * reB[j * n];
                for (let i = 1;i < n;i++) {
                    accRe += reA[i] * reB[j * n + i] + imA[0] * imB[j * n + i];
                    accIm += - reA[i] * imB[j * n + i] + imA[0] * reB[j * n + i];
                }
            }
        }
    }

    private _cmmul_mm(dims: [number, number, number], opB: number, reA: DataBlock, imA: DataBlock,
                      reB: DataBlock, imB: DataBlock, reC: DataBlock, imC: DataBlock): void {
        let [m, n, p] = dims;
        if (opB === 0) {
            // A*B where, A: m x n, B: n x p
            let columnCacheRe: DataBlock = p === 1 ? reB : new Array(n);
            let columnCacheIm: DataBlock = p === 1 ? imB : new Array(n);
            for (let j = 0;j < p;j++) {
                // cache j-th column of B
                if (p !== 1) {
                    for (let k = 0;k < n;k++) {
                        columnCacheRe[k] = reB[k * p + j];
                        columnCacheIm[k] = imB[k * p + j];
                    }
                }
                // evaluate j-th column of C
                for (let i = 0;i < m;i++) {
                    let accRe = reA[i * n] * columnCacheRe[0] - imA[i * n] * columnCacheIm[0];
                    let accIm = reA[i * n] * columnCacheIm[0] + imA[i * n] * columnCacheRe[0];
                    for (let k = 1;k < n;k++) {
                        accRe += reA[i * n + k] * columnCacheRe[k] - imA[i * n + k] * columnCacheIm[k];
                        accIm += reA[i * n + k] * columnCacheIm[k] + imA[i * n + k] * columnCacheRe[k];
                    }
                    reC[i * p + j] = accRe;
                    imC[i * p + j] = accIm;
                }
            }
        } else if (opB === 1) {
            // A*B^T where, A: m x n, B: p x n
            for (let i = 0;i < m;i++) {
                // evaluate j-th column of C
                for (let j = 0;j < p;j++) {
                    let accRe = reA[i * n] * reB[j * n] - imA[i * n] * imB[j * n];
                    let accIm = reA[i * n] * imB[j * n] + imA[i * n] * reB[j * n];
                    for (let k = 1;k < n;k++) {
                        accRe += reA[i * n + k] * reB[j * n + k] - imA[i * n + k] * imB[j * n + k];
                        accIm += reA[i * n + k] * imB[j * n + k] + imA[i * n + k] * reB[j * n + k];
                    }
                    reC[i * p + j] = accRe;
                    imC[i * p + j] = accIm;
                }
            }
        } else {
            // A*B^H where, A: m x n, B: p x n
            for (let i = 0;i < m;i++) {
                // evaluate j-th column of C
                for (let j = 0;j < p;j++) {
                    let accRe = reA[i * n] * reB[j * n] + imA[i * n] * imB[j * n];
                    let accIm = - reA[i * n] * imB[j * n] + imA[i * n] * reB[j * n];
                    for (let k = 1;k < n;k++) {
                        accRe += reA[i * n + k] * reB[j * n + k] + imA[i * n + k] * imB[j * n + k];
                        accIm += - reA[i * n + k] * imB[j * n + k] + imA[i * n + k] * reB[j * n + k];
                    }
                    reC[i * p + j] = accRe;
                    imC[i * p + j] = accIm;
                }
            }
        }
    }

    public transpose(dims: [number, number], A: DataBlock, B: DataBlock): void {
        if (B === A) {
            throw new Error('In-place transpose is not supported.');
        }
        var nr = dims[0],
            nc = dims[1],
            blockSize = this._chunkSize;
        for (let ii = 0; ii < nr; ii += blockSize) {
            for (let jj = 0; jj < nc; jj += blockSize) {
                let iMax = ii + Math.min(blockSize, nr - ii);
                let jMax = jj + Math.min(blockSize, nc - jj);
                for (let i = ii; i < iMax; i++) {
                    for (let j = jj; j < jMax; j++) {
                        B[j * nr + i] = A[i * nc + j];
                    }
                }
            }
        }
    }
    
    public hermitian(dims: [number, number], reA: DataBlock, imA: DataBlock, reB: DataBlock, imB: DataBlock): void {
        let nr = dims[0];
        let nc = dims[1];
        let blockSize = this._chunkSize;
        // real part
        this.transpose(dims, reA, reB);
        // imaginary part
        if (imA === imB) {
            throw new Error('In-place Hermitian is not supported.');
        }
        for (let ii = 0; ii < nr; ii += blockSize) {
            for (let jj = 0; jj < nc; jj += blockSize) {
                let iMax = ii + Math.min(blockSize, nr - ii);
                let jMax = jj + Math.min(blockSize, nc - jj);
                for (let i = ii; i < iMax; i++) {
                    for (let j = jj; j < jMax; j++) {
                        imB[nr * j + i] = -imA[nc * i + j];
                    }
                }
            }
        }
    }
}
