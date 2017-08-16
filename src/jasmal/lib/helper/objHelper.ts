import { TypedArray } from '../commonTypes';

export class ExtendChain<S> {

    private _obj: S;

    constructor(src: S) {
        this._obj = src;
    }
    
    public extend<E>(ext: E): ExtendChain<S & E> {
        return new ExtendChain(ObjectHelper.extend(this._obj, ext));
    }

    public end(): S {
        return this._obj;
    }

}

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

    
    public static extend<S, E>(src: S, ext: E): S & E {
        for (let prop in ext) {
            if (ext.hasOwnProperty(prop)) {
                (<any>src)[prop] = ext[prop];
            }
        }
        return <S & E>src;
    }

    public static createExtendChain<S>(src: S): ExtendChain<S> {
        return new ExtendChain(src);
    }

    public static properties(obj: object): string[] {
        let result: string[] = [];
        for (let prop in obj) {
            if (obj.hasOwnProperty(prop)) {
                result.push(prop);
            }
        }
        return result;
    }

}