import { DataBlock } from '../../commonTypes';

export interface IMatrixTransposeBackend {

    transpose(dims: [number, number], A: ArrayLike<number>, B: DataBlock): void;
    
    hermitian(dims: [number, number], reA: ArrayLike<number>, imA: ArrayLike<number>,
              reB: DataBlock, imB: DataBlock): void;

}

export class BuiltInMTB implements IMatrixTransposeBackend {

    private _chunkSize: number;
    
        constructor(chunkSize: number = 32) {
            this._chunkSize = chunkSize;
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
