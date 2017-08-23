import { ComplexNumber } from '../lib/complexNumber';
import { Tensor } from '../lib/tensor';
import { ShapeHelper } from '../lib/helper/shapeHelper';
import { DTypeHelper } from '../lib/dtype';

function areCloseByAbsoluteValue(x: number, y: number, tol: number): boolean {
    return Math.abs(x - y) <= tol;
}

function areCloseByPrecision(x: number, y: number, n: number): boolean {
    let xStr = x.toExponential(n);
    let yStr = y.toExponential(n);
    return xStr === yStr;
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

export function checkNumber(actual: any, expected: number, tolerance: number = 0): void {
    if (actual === expect) {
        return;
    }
    if (typeof actual !== 'number') {
        fail(`Expecting a number got "O=${Object.prototype.toString.call(actual)}".`);
        return;
    }
    if (isNaN(expected)) {
        if (!isNaN(actual)) {
            fail(`Expecting a NaN, got ${actual}.`);
            return;
        }
    } else {
        if (isNaN(actual)) {
            fail('Expecting a valid number, got a NaN.');
            return;
        } else if (!areCloseByAbsoluteValue(actual, expected, tolerance)) {
            fail(`Expecting ${expected} ± ${tolerance}, got ${actual}.`);
            return;
        }
    }
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

export function checkComplex(actual: any, expected: ComplexNumber, tolerance: number = 0): void {
    if (actual === expected) return;
    if (!(actual instanceof ComplexNumber)) {
        throw new Error('Expecting a complex number.');
    }
    if (!areCloseByAbsoluteValue(actual.re, expected.re, tolerance)) {
        fail(`Expecting the real part to be ${expected.re} ± ${tolerance}, got ${actual.re}.`);
        return;
    }
    if (!areCloseByAbsoluteValue(actual.im, expected.im, tolerance)) {
        fail(`Expecting the real part to be ${expected.im} ± ${tolerance}, got ${actual.im}.`);
        return;
    }
}

export function checkTensor(actual: any, expected: Tensor, tolerance: number = 0, absolute: boolean = true): void {
    if (actual === expected) return;
    if (!(actual instanceof Tensor)) {
        fail(`Expecting a tensor, but got "${Object.prototype.toString.call(actual)}".`);
        return;
    }
    if (!ShapeHelper.compareShape(actual.shape, expected.shape)) {
        fail(`Expected shape: ${ShapeHelper.shapeToString(expected.shape)}, ` +
            `actual shape: ${ShapeHelper.shapeToString(actual.shape)}.`);
        return;
    }
    if (actual.dtype !== expected.dtype) {
        fail(`Expected dtype is ${DTypeHelper.dTypeToString(expected.dtype)}, ` +
            `actual dtype is ${DTypeHelper.dTypeToString(actual.dtype)}.`);
        return;
    }
    if (expected.hasComplexStorage() && !actual.hasComplexStorage()) {
        fail('Expecting a tensor with complex storage, got a tensor without one.');
        return;
    }
    if (!expected.hasComplexStorage() && actual.hasComplexStorage()) {
        fail('Expecting a tensor without complex storage, got a tensor with one.');
        return;
    }
    let reActual = actual.realData;
    let reExpected = expected.realData;
    for (let i = 0;i < reActual.length;i++) {
        if (isNaN(reExpected[i])) {
            if (!isNaN(reActual[i])) {
                fail(`At index ${i}, expecting the real part to be NaN, got ${reActual[i]}.`);
                return;
            }
        } else if (!isFinite(reExpected[i])) {
            if (reExpected[i] !== reActual[i]) {
                fail(`At index ${i}, expecting the real part to be ${reExpected[i]}, got ${reActual[i]}.`);
                return;                
            }
        } else {
            if (isNaN(reActual[i])) {
                fail(`At index ${i}, expecting the real part to be a valid number, got NaN.`);
                return;
            } else {
                if (absolute) {
                    if (!areCloseByAbsoluteValue(reActual[i], reExpected[i], tolerance)) {
                        fail(`At index ${i}, expecting the real part to be ${reExpected[i]} ± ${tolerance}, got ${reActual[i]}.`);
                        return;
                    }
                } else {
                    if (!areCloseByPrecision(reActual[i], reExpected[i], tolerance)) {
                        fail(`At index ${i}, expecting the real part to be close to ${reExpected[i].toExponential(tolerance)}` +
                            ` (actual value is ${reExpected[i]}), got ${reActual[i].toExponential(tolerance)}.`);
                        return;
                    }
                }                
            }
        }
    }
    if (expected.hasComplexStorage()) {
        let imActual = actual.imagData;
        let imExpected = expected.imagData;
        for (let i = 0;i < imActual.length;i++) {
            if (isNaN(imExpected[i])) {
                if (!isNaN(imActual[i])) {
                    fail(`At index ${i}, expecting the imaginary part to be NaN, got ${imActual[i]}.`);
                    return;
                }
            } else if (!isFinite(imExpected[i])) {
                if (imExpected[i] !== imActual[i]) {
                    fail(`At index ${i}, expecting the imaginary part to be ${imExpected[i]}, got ${imActual[i]}.`);
                    return;                
                }
            } else {
                if (isNaN(imActual[i])) {
                    fail(`At index ${i}, expecting the imaginary part to be a valid number, got NaN.`);
                    return;
                } else {
                    if (absolute) {
                        if (!areCloseByAbsoluteValue(imActual[i], imExpected[i], tolerance)) {
                            fail(`At index ${i}, expecting the imaginary part to be ${reExpected[i]} ± ${tolerance}, got ${reActual[i]}.`);
                            return;
                        }
                    } else {
                        if (!areCloseByPrecision(imActual[i], imExpected[i], tolerance)) {
                            fail(`At index ${i}, expecting the imaginary part to be close to ${reExpected[i].toExponential(tolerance)}` +
                                ` (actual value is ${reExpected[i]}), got ${reActual[i].toExponential(tolerance)}.`);
                            return;
                        }
                    }
                }
            }
        }
    }
}