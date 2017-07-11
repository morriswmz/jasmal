import { TypedArray } from '../commonTypes';

export class ObjectHelper {
    public static isTypedArray(x: any): x is TypedArray {
        return (x instanceof Float64Array) ||
            (x instanceof Float32Array) || 
            (x instanceof Int32Array) ||
            (x instanceof Int16Array) ||
            (x instanceof Int8Array) ||
            (x instanceof Uint8Array) ||
            (x instanceof Uint16Array) ||
            (x instanceof Uint32Array) ||
            (x instanceof Uint8ClampedArray);
    }
}