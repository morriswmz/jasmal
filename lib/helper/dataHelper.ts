import { DataBlock } from '../storage';

export class DataHelper {

    public static areArraysEqual(x: ArrayLike<number>, y: ArrayLike<number>): boolean {
        if (x === y) return true;
        if (x.length !== y.length) return false;
        for (let i = 0;i < x.length;i++) {
            if (x[i] !== y[i]) {
                return false;
            }
        }
        return true;
    }

    public static areArraysApproximatelyEqual(x: ArrayLike<number>,
                                              y: ArrayLike<number>,
                                              tolerance: number): boolean {
        if (x === y) return true;
        if (x.length !== y.length) return false;
        for (let i = 0;i < x.length;i++) {
            if (Math.abs(x[i] - y[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    public static isArrayAllNonZeros(x: ArrayLike<number>): boolean {
        for (let i = 0;i < x.length;i++) {
            if (x[i] === 0) {
                return false;
            }
        }
        return true;
    }

    public static isArrayAllZeros(x: ArrayLike<number>): boolean {
        for (let i = 0;i < x.length;i++) {
            if (x[i] !== 0) {
                return false;
            }
        }
        return true;
    }

    public static isArrayApproximatelyAllZeros(x: ArrayLike<number>,
                                               tolerance: number): boolean {
        for (let i = 0;i < x.length;i++) {
            if (Math.abs(x[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    public static findReal(re: ArrayLike<number>): number[] {
        let indices: number[] = [];
        for (let i = 0;i < re.length;i++) {
            if (re[i] !== 0) {
                indices.push(i);
            }
        }
        return indices;
    }

    public static findComplex(re: ArrayLike<number>, im: ArrayLike<number>): number[] {
        let indices: number[] = [];
        for (let i = 0;i < re.length;i++) {
            if (re[i] !== 0 || im[i] !== 0) {
                indices.push(i);
            }
        }
        return indices;
    }

    public static findWithCallbackReal(re: ArrayLike<number>, cb: (re: number, im?: number) => boolean): number[] {
        let indices: number[] = [];
        for (let i = 0;i < re.length;i++) {
            if (cb(re[i], 0)) {
                indices.push(i);
            }
        }
        return indices;
    }

    public static findWithCallbackComplex(re: ArrayLike<number>,
                                          im: ArrayLike<number>,
                                          cb: (re: number, im?: number) => boolean): number[] {
        let indices: number[] = [];
        for (let i = 0;i < re.length;i++) {
            if (cb(re[i], im[i])) {
                indices.push(i);
            }
        }
        return indices;
    }
    
    public static copy(from: ArrayLike<number>, to: DataBlock, offset: number = 0) {
        if (from.length + offset > to.length) {
            throw new Error('Not enough space.');
        }
        for (let i = 0;i < from.length;i++) {
            to[i + offset] = from[i];
        }
    }

}