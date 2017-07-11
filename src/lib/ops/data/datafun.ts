export class DataFunction {
    
    /**
     * Finds the maximum element and its index. If the maximum is attained by
     * more than one element, the index of the first element that attains the
     * maximum is returned.
     * @param x 
     */
    public static max(x: ArrayLike<number>): [number, number];
    public static max(x: ArrayLike<number>, offset: number, stride: number, n: number): [number, number];
    public static max(x: ArrayLike<number>, offset: number = 0, stride: number = 1, n: number = -1): [number, number] {
        let ub: number;
        [n, ub] = DataFunction._processArgs(x.length, offset, stride, n);
        let max = -Infinity, idx = offset;
        for (let i = offset;i < ub;i += stride) {
            if (x[i] > max) {
                max = x[i];
                idx = i;
            }
        }
        return [max, idx];
    }

    /**
     * Finds the minimum element and its index. If the minimum is attained by
     * more than one element, the index of the first element that attains the
     * minimum is returned.
     * @param x 
     */
    public static min(x: ArrayLike<number>): [number, number];
    public static min(x: ArrayLike<number>, offset: number, stride: number, n: number): [number, number];
    public static min(x: ArrayLike<number>, offset: number = 0, stride: number = 1, n: number = -1): [number, number] {
        let ub: number;
        [n, ub] = DataFunction._processArgs(x.length, offset, stride, n);
        let min = Infinity, idx = offset;
        for (let i = offset;i < ub;i += stride) {
            if (x[i] < min) {
                min = x[i];
                idx = i;
            }
        }
        return [min, idx];
    }

    public static sum(x: ArrayLike<number>): number;
    public static sum(x: ArrayLike<number>, offset: number, stride: number, n: number): number;
    public static sum(x: ArrayLike<number>, offset: number = 0, stride: number = 1, n: number = -1): number {
        let ub: number;
        [n, ub] = DataFunction._processArgs(x.length, offset, stride, n);
        let acc = 0;      
        for (let i = offset;i < ub;i += stride) {
            acc += x[i];
        }
        return acc;
    }

    public static prod(x: ArrayLike<number>): number;
    public static prod(x: ArrayLike<number>, offset: number, stride: number, n: number): number;
    public static prod(x: ArrayLike<number>, offset: number = 0, stride: number = 1, n: number = -1): number {
        let ub: number;
        [n, ub] = DataFunction._processArgs(x.length, offset, stride, n);
        let acc = 1;
        for (let i = offset;i < ub;i += stride) {
            acc *= x[i];
            if (acc === 0) {
                break;
            }
        }
        return acc;
    }

    public static cprod(reX: ArrayLike<number>, imX: ArrayLike<number>): [number, number];
    public static cprod(reX: ArrayLike<number>, imX: ArrayLike<number>,
                        offset: number, stride: number, n: number): [number, number];
    public static cprod(reX: ArrayLike<number>, imX: ArrayLike<number>,
                        offset: number = 0, stride: number = 1, n: number = -1): [number, number] {
        let ub: number;
        [n, ub] = DataFunction._processArgs(reX.length, offset, stride, n);
        let accRe = 1;
        let accIm = 0;    
        for (let i = offset;i < ub;i += stride) {
            let tmp = accRe;
            accRe = tmp * reX[i] - accIm * imX[i];
            accIm = tmp * imX[i] + accIm * reX[i];
        }
        return [accRe, accIm];
    }

    public static var(x: ArrayLike<number>): number;
    public static var(x: ArrayLike<number>, offset: number, stride: number, n: number): number;
    public static var(x: ArrayLike<number>, offset: number = 0, stride: number = 1, n: number = -1): number {
        let ub: number;
        [n, ub] = DataFunction._processArgs(x.length, offset, stride, n);
        if (n === 1) {
            return isFinite(x[offset]) ? 0 : NaN;
        }
        let u = 0;
        for (let i = offset;i < ub;i += stride) {
            if (!isFinite(x[i])) {
                return NaN;
            }
            u += x[i];
        }
        u /= n;
        let v = 0;
        for (let i = offset;i < ub;i += stride) {
            v += (x[i] - u) * (x[i] - u);
        }
        return v / (n - 1);
    }

    public static cvar(reX: ArrayLike<number>, imX: ArrayLike<number>): number;
    public static cvar(reX: ArrayLike<number>, imX: ArrayLike<number>, offset: number, stride: number, n: number): number;
    public static cvar(reX: ArrayLike<number>, imX: ArrayLike<number>, offset: number = 0, stride: number = 1, n: number = -1): number {
        let ub: number;
        [n, ub] = DataFunction._processArgs(reX.length, offset, stride, n);
        if (n === 1) {
            return (isFinite(reX[offset]) && isFinite(imX[offset])) ? 0 : NaN;
        }
        let uRe = 0, uIm = 0;
        for (let i = offset;i < ub;i += stride) {
            if (!isFinite(reX[offset]) || !isFinite(imX[offset])) {
                return NaN;
            }
            uRe += reX[i];
            uIm += imX[i];
        }
        uRe /= n;
        uIm /= n;
        let v = 0;
        for (let i = offset;i < ub;i += stride) {
            v += (reX[i] - uRe) * (reX[i] - uRe) + (imX[i] - uIm) * (imX[i] - uIm);
        }
        return v / (n - 1);
    }

    public static median(reX: ArrayLike<number>);
    public static median(reX: ArrayLike<number>, offset: number, stride: number, n: number): number;
    public static median(reX: ArrayLike<number>, offset: number = 0, stride: number = 1, n: number = -1): number {
        let ub: number;
        [n, ub] = DataFunction._processArgs(reX.length, offset, stride, n);
        if (n === 1) {
            return reX[offset];
        }
        let arr = new Array(n);
        for (let i = 0;i < n;i++) {
            arr[i] = reX[offset + i * stride];
        }
        arr.sort();
        // check for NaNs
        if (isNaN(arr[n - 1])) {
            return NaN;
        }
        if (n % 2 === 0) {
            // even number of elements
            return 0.5 * (arr[n / 2] + arr[n / 2 - 1]);
        } else {
            // odd number of elements
            return arr[(n - 1) / 2];
        }
    }

    private static _processArgs(arrLength: number, offset: number, stride: number, n: number): [number, number] {
        if (arrLength === 0) {
            throw new Error('Array cannot be empty.')
        }
        let ub = n < 0 ? arrLength : offset + n * stride;
        if (ub >= arrLength + stride) {
            throw new Error('Maximum index is out of bounds.');
        }
        return [n < 0 ? arrLength : n, ub];
    }


}