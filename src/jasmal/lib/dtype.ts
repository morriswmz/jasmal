import { ObjectHelper } from './helper/objHelper';

/**
 * Data type.
 */
export enum DType {
    LOGIC = 0,
    INT32 = 1,
    FLOAT64 = 3
}

export class DTypeHelper {
    /**
     * Gets the string representation of the given data type.
     * @param dtype Data type
     */
    public static dTypeToString(dtype: DType): string {
        switch (dtype) {
            case DType.LOGIC: return 'logic';
            case DType.INT32: return 'int32';
            case DType.FLOAT64: return 'float64';
            default: return 'unknown';
        }
    }

    /**
     * Checks if the new type is wider than the original type.
     * FLOAT64 > INT32 > LOGIC
     * @param original The original data type.
     * @param newType The new data type.
     */
    public static isWiderType(original: DType, newType: DType): boolean {
        return newType > original;
    }

    /**
     * Returns the wider type between the two.
     * @param t1 DType 1.
     * @param t2 DType 2.
     */
    public static getWiderType(t1: DType, t2: DType): DType {
        return t1 > t2 ? t1 : t2;
    }

    public static getDTypeOfIndices(): DType {
        return ObjectHelper.hasTypedArraySupport() ? DType.INT32 : DType.FLOAT64;
    }

}

export class OutputDTypeResolver {
    
    /**
     * Returns the same data type as that of the input.
     * @param t Data type of the input.
     */
    public static uNoChange(t: DType): DType {
        return t;
    }

    /**
     * Always returns LOGIC.
     */
    public static uToLogic(): DType {
        return DType.LOGIC;
    }

    /**
     * Returns INT32.
     */
    public static uToInt32(): DType {
        return DType.INT32;
    }

    /**
     * Returns FLOAT64.
     */
    public static uToFloat(): DType {
        return DType.FLOAT64;
    }

    /**
     * Always returns LOGIC if the input is real.
     * Otherwise undefined is returned.
     * @param _t 
     * @param isComplex 
     */
    public static uToLogicRealOnly(_t: DType, isComplex: boolean): DType | undefined {
        return isComplex ? undefined : DType.LOGIC;
    }

    /**
     * Returns FLOAT64 only when the input type is LOGIC.
     * Otherwise returns the input's data type.
     * @param t 
     * @param _isComplex 
     */
    public static uOnlyLogicToFloat(t: DType, _isComplex: boolean): DType {
        return t === DType.LOGIC ? DType.FLOAT64 : t;
    }

    /**
     * Returns INT32 when the input data type LOGIC. Otherwise, returns the
     * original data type.
     * @param t 
     */
    public static uNoChangeExceptLogicToInt(t: DType): DType {
        return t === DType.LOGIC ? DType.INT32 : t;
    }

    /**
     * Returns the wider data type between the two inputs.
     * @param t1 
     * @param _isComplex1 
     * @param t2 
     * @param _isComplex2 
     */
    public static bWider(t1: DType, _isComplex1: boolean, t2: DType, _isComplex2: boolean): DType {
        return DTypeHelper.getWiderType(t1, t2);
    }

    /**
     * Converts LOGIC to INT32 first, and then returns the wider type. 
     * @param t1 
     * @param _isComplex1 
     * @param t2 
     * @param _isComplex2 
     */
    public static bWiderWithLogicToInt(t1: DType, _isComplex1: boolean, t2: DType, _isComplex2: boolean): DType {
        return DTypeHelper.getWiderType(
            t1 === DType.LOGIC ? DType.INT32 : t1,
            t2 === DType.LOGIC ? DType.INT32 : t2);
    }

    /**
     * Returns INT32.
     */
    public static bToInt32(): DType {
        return DType.INT32;
    }

    /**
     * Returns FLOAT64.
     */
    public static bToFloat(): DType {
        return DType.FLOAT64;
    }

    /**
     * Always returns LOGIC.
     */
    public static bToLogic(): DType {
        return DType.LOGIC;
    }

    /**
     * Always returns LOGIC when both inputs are real.
     * Returns undefined when any of the inputs is complex.
     * @param _t1 
     * @param isComplex1 
     * @param _t2 
     * @param isComplex2 
     */
    public static bToLogicRealOnly(_t1: DType, isComplex1: boolean, _t2: DType, isComplex2: boolean): DType | undefined {
        return (isComplex1 || isComplex2) ? undefined : DType.LOGIC;
    }

}
