import { SVD } from './decomp/svd';
import { MathHelper } from '../../helper/mathHelper';

export class NormFunction {

    public static vec0Norm(reX: ArrayLike<number>): number {
        let norm = 0;
        for (let i = 0;i < reX.length;i++) {
            if (isNaN(reX[i])) {
                return NaN;
            }
            if (reX[i] !== 0) {
                norm++;
            }
        }
        return norm;
    }

    public static cvec0Norm(reX: ArrayLike<number>, imX: ArrayLike<number>): number {
        let norm = 0;
        for (let i = 0;i < reX.length;i++) {
            if (isNaN(reX[i]) || isNaN(imX[i])) {
                return NaN;
            }
            if (reX[i] !== 0 || imX[i] !== 0) {
                norm++;
            }
        }
        return norm;
    }

    public static vec2Norm(reX: ArrayLike<number>): number {
        let norm = 0;
        for (let i = 0;i < reX.length;i++) {
            if (isNaN(reX[i])) {
                return NaN;
            }
            norm += reX[i] * reX[i];
        }
        return Math.sqrt(norm);
    }

    public static cvec2Norm(reX: ArrayLike<number>, imX: ArrayLike<number>): number {
        let norm = 0;
        for (let i = 0;i < reX.length;i++) {
            if (isNaN(reX[i]) || isNaN(imX[i])) {
                return NaN;
            }
            norm += reX[i] * reX[i] + imX[i] * imX[i];
        }
        return Math.sqrt(norm);
    }

    public static vecInfNorm(reX: ArrayLike<number>): number {
        let norm = 0;
        for (let i = 0;i < reX.length;i++) {
            if (isNaN(reX[i])) {
                return NaN;
            }
            let v = Math.abs(reX[i]);
            if (v > norm) {
                norm = v;
            }
        }
        return norm;
    }

    public static cvecInfNorm(reX: ArrayLike<number>, imX: ArrayLike<number>): number {
        let norm = 0;
        for (let i = 0;i < reX.length;i++) {
            if (isNaN(reX[i]) || isNaN(imX[i])) {
                return NaN;
            }
            let v = MathHelper.length2(reX[i], imX[i]);
            if (v > norm) {
                norm = v;
            }
        }
        return norm;
    }

    public static vecPNorm(reX: ArrayLike<number>, p: number): number {
        let norm = 0;
        for (let i = 0;i < reX.length;i++) {
            if (isNaN(reX[i])) {
                return NaN;
            }
            norm += Math.pow(Math.abs(reX[i]), p);
        }
        return Math.pow(norm, 1/p);
    }

    public static cvecPNorm(reX: ArrayLike<number>, imX: ArrayLike<number>, p: number): number {
        let norm = 0;
        for (let i = 0;i < reX.length;i++) {
            if (isNaN(reX[i]) || isNaN(imX[i])) {
                return NaN;
            }
            norm += Math.pow(MathHelper.length2(reX[i], imX[i]), p);
        }
        return Math.pow(norm, 1/p);
    }

    public static mat1Norm(m: number, n: number, reX: ArrayLike<number>): number {
        let norm = 0, colSum: number, el: number;
        for (let j = 0;j < n;j++) {
            colSum = 0;
            for (let i = 0;i < m;i++) {
                el = reX[i * n + j];
                if (isNaN(el)) {
                    return NaN;
                }
                colSum += Math.abs(el);
            }
            if (colSum > norm) {
                norm = colSum;
            }
        }
        return norm;
    }

    public static cmat1Norm(m: number, n: number, reX: ArrayLike<number>, imX: ArrayLike<number>): number {
        let norm = 0, colSum: number, elRe: number, elIm: number;
        for (let j = 0;j < n;j++) {
            colSum = 0;
            for (let i = 0;i < m;i++) {
                elRe = reX[i * n + j];
                elIm = reX[i * n + j];
                if (isNaN(elRe) || isNaN(elIm)) {
                    return NaN;
                }
                colSum += MathHelper.length2(elRe, elIm);
            }
            if (colSum > norm) {
                norm = colSum;
            }
        }
        return norm;
    }

    public static mat2Norm(m: number, n: number, reX: ArrayLike<number>): number {
        let s = new Array(n);
        // we temporarily use svd here
        SVD.svd(m, n, false, reX, s, []);
        return s[0];
    }

    public static cmat2Norm(m: number, n: number, reX: ArrayLike<number>, imX: ArrayLike<number>): number {
        let s = new Array(n);
        // we temporarily use svd here
        SVD.csvd(m, n, false, reX, imX, s, [], []);
        return s[0];
    }

    public static matInfNorm(m: number, n: number, reX: ArrayLike<number>): number {
        let norm = 0, rowSum: number, el: number;
        for (let i = 0;i < m;i++) {
            rowSum = 0;
            for (let j = 0;j < n;j++) {
                el = reX[i * n + j];
                if (isNaN(el)) {
                    return NaN;
                }
                rowSum += Math.abs(el);
            }
            if (rowSum > norm) {
                norm = rowSum;
            }
        }
        return norm;
    }

    public static cmatInfNorm(m: number, n: number, reX: ArrayLike<number>, imX: ArrayLike<number>): number {
        let norm = 0, rowSum: number, elRe: number, elIm: number;
        for (let i = 0;i < m;i++) {
            rowSum = 0;
            for (let j = 0;j < n;j++) {
                elRe = reX[i * n + j];
                elIm = reX[i * n + j];
                if (isNaN(elRe) || isNaN(elIm)) {
                    return NaN;
                }
                rowSum += MathHelper.length2(elRe, elIm);
            }
            if (rowSum > norm) {
                norm = rowSum;
            }
        }
        return norm;
    }

}