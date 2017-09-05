import { TypedArray } from '../commonTypes';
import { ObjectHelper } from './objHelper';

export interface BroadcastingCheckResult {
    /**
     * Adjusted (by prepending new axis) shape of the first operand.
     */
    shapeX: ArrayLike<number>;
    /**
     * Adjusted (by prepending new axis) shape of the second operand.
     */
    shapeY: ArrayLike<number>;
    /**
     * Shape of the resulting tensor.
     */
    shapeZ: ArrayLike<number>;
    /**
     * True if the two operands share the exact shape.
     */
    exact: boolean;
}

export class ShapeHelper {

    public static validateShape(shape: ArrayLike<number>): void {
        if (shape.length === 0 || shape.length == undefined) {
            throw new Error('Shape must be a non-empty array.');
        }
        for (let i = 0;i < shape.length;i++) {
            if ((shape[i] | 0) !== shape[i]) {
                throw new Error('Shape can only consists of integers.');
            }
            if (shape[i] <= 0) {
                throw new Error(`The length of dimension ${i} must be positive.`);
            }
        }
    }

    public static getSizeFromShape(shape: ArrayLike<number>): number {
        let s = shape[0];
        for (let i = 1;i < shape.length;i++) {
            s *= shape[i];
        }
        return s;
    }

    public static computeStrides(shape: ArrayLike<number>): number[] {
        let strides = [1];
        let d = 1;
        for (let i = shape.length - 1;i > 0;i--) {
            d *= shape[i];
            strides.unshift(d);
        }
        return strides;
    }

    public static shapeToString(shape: ArrayLike<number>): string {
        if (Array.isArray(shape)) {
            return `[${shape.join('x')}]`;
        } else {
            return `[${Array.prototype.join.call(shape, 'x')}]`;
        }
    }

    public static isScalarShape(shape: ArrayLike<number>): boolean {
        for (let i = 0;i < shape.length;i++) {
            if (shape[i] !== 1) {
                return false;
            }
        }
        return true;
    }

    public static getSqueezedShape(shape: ArrayLike<number>): number[] {
        let newShape: number[] = [];
        for (let i = 0;i < shape.length;i++) {
            if (shape[i] !== 1) {
                newShape.push(shape[i]);
            }
        }
        if (newShape.length === 0) {
            newShape.push(1);
        }
        return newShape;
    }

    public static compareShape(shape1: ArrayLike<number>, shape2: ArrayLike<number>): boolean {
        if (shape1.length !== shape2.length) {
            return false;
        }
        for (let i = 0;i < shape1.length;i++) {
            if (shape1[i] !== shape2[i]) {
                return false;
            }
        }
        return true;
    }

    public static compareSqueezedShape(shape1: ArrayLike<number>, shape2: ArrayLike<number>): boolean {
        return ShapeHelper.compareShape(ShapeHelper.getSqueezedShape(shape1), ShapeHelper.getSqueezedShape(shape2));
    }

    /**
    * Checks if the broadcasting is possible between the two shapes.
    * @param shapeX Shape of tensor X.
    * @param shapeY Shape of tensor Y.
    */
    public static checkBroadcastingCompatibility(shapeXIn: ArrayLike<number>, shapeYIn: ArrayLike<number>): BroadcastingCheckResult {
        'use strict';
        // check shape
        let shapeZ: number[] = [];
        if (shapeXIn.length < shapeYIn.length) {
            // need to modify shapeX, make a copy
            let shapeX = Array.prototype.slice.call(shapeXIn);
            while (shapeX.length < shapeYIn.length) shapeX.unshift(1);
            shapeXIn = shapeX;
        }
        if (shapeYIn.length < shapeXIn.length) {
            // need to modify shapeY, make a copy
            let shapeY = Array.prototype.slice.call(shapeYIn);
            while (shapeY.length < shapeXIn.length) shapeY.unshift(1);
            shapeYIn = shapeY;
        }
        let exact = true;
        for (let i = 0;i < shapeXIn.length;i++) {
            if (shapeXIn[i] !== shapeYIn[i]) {
                if (shapeXIn[i] !== 1 && shapeYIn[i] !== 1) {
                    throw new Error('Incompatible shape.')
                }
                exact = false;
            }
            shapeZ.push(Math.max(shapeXIn[i], shapeYIn[i]));
        }
        return {
            shapeX: shapeXIn,
            shapeY: shapeYIn,
            shapeZ: shapeZ,
            exact: exact
        };
    }

    /**
     * Infers the shape of a (nested) JavaScript array (e.g., the shape of
     * `[[1, 2], [3, 5], [-1, 2]]` is `[3, 2]`).
     * @param arr A (nested) JavaScript array.
     */
    public static inferShapeFromArray(arr: any[] | TypedArray): number[] {
        let shape: number[] = [];
        let curEl: any = arr;
        while (Array.isArray(curEl) || ObjectHelper.isTypedArray(curEl)) {
            if (curEl.length === 0) {
                throw new Error('Array cannot be empty.');
            }
            shape.push(curEl.length);
            curEl = curEl[0];
        }
        if (shape.length === 0) {
            throw new Error('Input is not an array.');
        }
        return shape;
    }

    /**
     * Validates if the given JavaScript array has the given shape.
     * @param arr The JavaScript array to be validated.
     * @param shape Expected shape.
     * @param level Used for recursive calling.
     */
    public static validateArrayShape(arr: any[] | TypedArray, shape: number[], level: number = 0): void {
        if (arr.length !== shape[level]) {
            throw new Error('The structure of the input array does not match that of a tensor.');
        }
        if (level < shape.length - 1) {
            for (let i = 0;i < arr.length;i++) {
                if (Array.isArray(arr[i])) {
                    ShapeHelper.validateArrayShape(arr[i], shape, level + 1);
                } else {
                    throw new Error('Cannot have mixed array and non-array elements at the same level.');
                }
            }
        }
    }

}
