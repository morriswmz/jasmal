import { ComplexNumber } from '../lib/core/complexNumber';
import { Tensor } from '../lib/core/tensor';
import { ShapeHelper } from '../lib/helper/shapeHelper';
import { DTypeHelper } from '../lib/core/dtype';
import { OpInput, OpOutput, Scalar } from '../lib/commonTypes';

function areCloseByAbsoluteValue(actual: number, expected: number, tol: number): boolean {
    return Math.abs(actual - expected) <= tol;
}

function areCloseByPrecision(actual: number, expected: number, n: number): boolean {
    if (expected === 0) {
        return Math.pow(10, n) * Math.abs(expected) < 1;
    } else {
        let xStr = actual.toExponential(n);
        let yStr = expected.toExponential(n);
        return xStr === yStr;
    }
}

export function maxAbs(x: ArrayLike<number>): number {
    let max = -Infinity;
    for (let i = 0;i < x.length;i++) {
        let abs = Math.abs(x[i]);
        if (abs > max) {
            max = abs;
        }
    }
    return max;
}

export function checkNumber(actual: any, expected: number, tolerance: number = 0,
                            absolute: boolean = true, msgPrefix: string = ''): boolean {
    if (actual === expect) {
        return true;
    }
    if (typeof actual !== 'number') {
        fail(`${msgPrefix}Expecting a number got "O=${Object.prototype.toString.call(actual)}".`);
        return false;
    }
    if (isNaN(expected)) {
        // NaN handling
        if (!isNaN(actual)) {
            fail(`${msgPrefix}Expecting a NaN, got ${actual}.`);
            return false;
        }
    } else {
        if (!isFinite(expected)) {
            // Infinity handling
            if (actual !== expected) {
                fail(`${msgPrefix}Expecting ${expected}, got ${actual}.`);
                return false;
            }
        } else {
            if (absolute) {
                if (!areCloseByAbsoluteValue(actual, expected, tolerance)) {
                    fail(`${msgPrefix}Expecting ${expected} ± ${tolerance}, got ${actual}.`);
                    return false;
                }
            } else {
                if (!areCloseByPrecision(actual, expected, tolerance)) {
                    fail(`${msgPrefix}Expecting the value to be close to ${expected.toExponential(tolerance)}` +
                    ` (actual value is ${expected}), got ${actual.toExponential(tolerance)}.`);
                    return false;
                }
            }
        }
    }
    return true;
}

export function checkArrayLike<T>(actual: ArrayLike<T>, expected: ArrayLike<T>): void {
    if (actual.length !== expected.length) {
        fail(`Expected an array with length ${expected.length}, but got an array with length ${actual.length}.`);
        return;
    }
    for (let i = 0;i < actual.length;i++) {
        if (actual[i] !== expected[i]) {
            fail(`expected[${i}] = ${expected[i]}, but actual[${i}] = ${actual[i]}.`);
            return;
        }
    }
};

export function checkComplex(actual: any, expected: ComplexNumber, tolerance: number = 0,
                             absolute: boolean = true, msgPrefix: string = ''): boolean {
    if (actual === expected) return true;
    if (!(actual instanceof ComplexNumber)) {
        throw new Error(`${msgPrefix}Expecting a complex number.`);
    }
    if (!checkNumber(actual.re, expected.re, tolerance, absolute, `${msgPrefix}Real part >> `)) {
        return false;
    }
    if (!checkNumber(actual.im, expected.im, tolerance, absolute, `${msgPrefix}Imaginary part >> `)) {
        return false;
    }
    return true;
}

export function checkTensor(actual: any, expected: Tensor, tolerance: number | ArrayLike<number> = 0, absolute: boolean = true): boolean {
    if (actual === expected) return true;
    if (!(actual instanceof Tensor)) {
        fail(`Expecting a tensor, but got "${Object.prototype.toString.call(actual)}".`);
        return false;
    }
    if (!ShapeHelper.compareShape(actual.shape, expected.shape)) {
        fail(`Expected shape: ${ShapeHelper.shapeToString(expected.shape)}, ` +
            `actual shape: ${ShapeHelper.shapeToString(actual.shape)}.`);
        return false;
    }
    if (actual.dtype !== expected.dtype) {
        fail(`Expected dtype is ${DTypeHelper.dTypeToString(expected.dtype)}, ` +
            `actual dtype is ${DTypeHelper.dTypeToString(actual.dtype)}.`);
        return false;
    }
    if (expected.hasComplexStorage() && !actual.hasComplexStorage()) {
        fail('Expecting a tensor with complex storage, got a tensor without one.');
        return false;
    }
    if (!expected.hasComplexStorage() && actual.hasComplexStorage()) {
        fail('Expecting a tensor without complex storage, got a tensor with one.');
        return false;
    }
    let isComplex = expected.hasComplexStorage();
    let reActual = actual.realData;
    let reExpected = expected.realData;
    let imActual = isComplex ? actual.imagData : [];
    let imExpected = isComplex ? expected.imagData : [];
    let pass: boolean = true;
    let i = 0, n = reExpected.length;
    if (typeof tolerance === 'number') {
        if (isComplex) {
            for (i = 0;i < n;i++) {
                pass = checkComplex(new ComplexNumber(reActual[i], imActual[i]),
                    new ComplexNumber(reExpected[i], imExpected[i]),
                    tolerance, absolute, `Index ${i} >> `);
                if (!pass) {
                    break;
                }
            }
        } else {
            for (i = 0;i < n;i++) {
                pass = checkNumber(reActual[i], reExpected[i], tolerance, absolute, `Index ${i} >> `);
                if (!pass) {
                    break;
                }
            }
        }
    } else {
        if (tolerance.length !== n) {
            throw new Error('The length of tolerance must match the size of the tensor.');
        }
        if (isComplex) {
            for (i = 0;i < n;i++) {
                pass = checkComplex(new ComplexNumber(reActual[i], imActual[i]),
                    new ComplexNumber(reExpected[i], imExpected[i]),
                    tolerance[i], absolute, `Index ${i} >> `);
                if (!pass) {
                    break;
                }
            }
        } else {
            for (i = 0;i < n;i++) {
                pass = checkNumber(reActual[i], reExpected[i], tolerance[i], absolute, `Index ${i} >> `);
                if (!pass) {
                    break;
                }
            }
        }
    }
    if (pass && i !== n) {
        throw new Error('Should have checked all the elements. However the loop index end with a incorrect value.');
    }
    return pass;
}

export function testUnaryOpInBatch(fn: (x: OpInput) => OpOutput, data: Array<[Scalar, Scalar, number]>, absolute: boolean = true) {
    let n = data.length;
    let reExpected = new Array(n);
    let imExpected: number[] = [];
    let reInput = new Array(n);
    let imInput: number[] = [];
    let tolArr: number[] = new Array(n);
    for (let i = 0;i < n;i++) {
        let input = data[i][0];
        let expected = data[i][1];
        if (input instanceof ComplexNumber) {
            if (imInput.length === 0) {
                imInput.length = n;
                for (let k = 0;k < n;k++) {
                    imInput[k] = 0;
                }
            }
            reInput[i] = input.re;
            imInput[i] = input.im;
        } else {
            reInput[i] = input;
        }
        if (expected instanceof ComplexNumber) {
            if (imExpected.length === 0) {
                imExpected.length = n;
                for (let k = 0;k < n;k++) {
                    imExpected[k] = 0;
                }
            }
            reExpected[i] = expected.re;
            imExpected[i] = expected.im;
        } else {
            reExpected[i] = expected;
        }
        tolArr[i] = data[i][2];
    }
    let actual = fn(Tensor.fromArray(reInput, imInput));
    let expected = Tensor.fromArray(reExpected, imExpected);
    checkTensor(actual, expected, tolArr, absolute);
}
