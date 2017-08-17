import { DataBlock } from '../commonTypes';
import { ObjectHelper } from './objHelper';

export class DataHelper {

    /**
     * Generates an array of natural numbers [0, 1, 2, ..., n].
     * @param n 
     */
    public static naturalNumbers(n: number): number[] {
        let s = new Array<number>(n);
        for (let i = 0;i < n;i++) {
            s[i] = i;
        }
        return s;
    }

    /**
     * Generates a typed array of natural numbers [0, 1, 2, ..., n].
     * @param n 
     */
    public static naturalNumbersAsInt32(n: number): DataBlock {
        if (ObjectHelper.hasTypedArraySupport()) {
            let s = new Int32Array(n);
            for (let i = 0;i < n;i++) {
                s[i] = i;
            }
            return s;
        } else {
            return DataHelper.naturalNumbers(n);
        }
    }

    public static allocateFloat64Array(size: number): DataBlock {
        return ObjectHelper.hasTypedArraySupport()
            ? new Float64Array(size)
            : DataHelper.allocateJsArray(size);
    }

    public static allocateInt32Array(size: number): DataBlock {
        return ObjectHelper.hasTypedArraySupport()
            ? new Int32Array(size)
            : DataHelper.allocateJsArray(size);
    }

    public static allocateJsArray(size: number): DataBlock {
        let arr = new Array<number>(size);
        for (let i = 0;i < size;i++) {
            arr[i] = 0;
        }
        return arr;
    }

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
            if (isNaN(x[i]) || isNaN(y[i]) || Math.abs(x[i] - y[i]) > tolerance) {
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
            if (isNaN(x[i]) || Math.abs(x[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    public static isArrayAllFinite(x: ArrayLike<number>): boolean {
        for (let i = 0;i < x.length;i++) {
            if (!isFinite(x[i])) {
                return false;
            }
        }
        return true;
    }

    public static anyNegative(x: ArrayLike<number>): boolean {
        for (let i = 0;i < x.length;i++) {
            if (x[i] < 0) {
                return true;
            }
        }
        return false;
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