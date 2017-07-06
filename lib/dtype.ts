// TODO: Remove FLOAT32 as JS does not support its operations
/**
 * Data type.
 */
export enum DType {
    LOGIC = 0,
    INT32 = 1,
    FLOAT32 = 2,
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
            case DType.FLOAT32: return 'float32';
            case DType.FLOAT64: return 'float64';
            default: return 'unknown';
        }
    }

    /**
     * Checks if the new type is wider than the original type.
     * FLOAT64 > FLOAT32 > INT32 > LOGIC
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

}

export class OutputDTypeCalculator {
    
    /**
     * Returns the same data type as that of the input.
     * @param t Data type of the input.
     * @param isComplex 
     */
    public static uNoChange(t: DType, isComplex: boolean): DType {
        return t;
    }

    /**
     * If the input's data type is FLOAT32, returns FLOAT32.
     * Otherwise returns FLOAT64.
     * @param t 
     * @param isComplex 
     */
    public static uToFloat(t: DType, isComplex: boolean): DType {
        return t === DType.FLOAT32 ? t : DType.FLOAT64;
    }

    /**
     * Always returns LOGIC if the input is real.
     * Otherwise undefined is returned.
     * @param t 
     * @param isComplex 
     */
    public static uToLogicRealOnly(t: DType, isComplex: boolean): DType | undefined {
        return isComplex ? undefined : DType.LOGIC;
    }

    /**
     * Returns FLOAT64 only when the input type is LOGIC.
     * Otherwise returns the input's data type.
     * @param t 
     * @param isComplex 
     */
    public static uOnlyLogicToFloat(t: DType, isComplex: boolean): DType {
        return t === DType.LOGIC ? DType.FLOAT64 : t;
    }

    /**
     * Returns the wider data type between the two inputs.
     * @param t1 
     * @param isComplex1 
     * @param t2 
     * @param isComplex2 
     */
    public static bWider(t1: DType, isComplex1: boolean, t2: DType, isComplex2: boolean): DType {
        return DTypeHelper.getWiderType(t1, t2);
    }

    /**
     * If any of the two inputs has a data type of FLOAT64, returns FLOAT64.
     * Otherwise, if any of the two inputs has a data type of FLOAT32, returns
     * FLOAT32.
     * Otherwise, returns FLOAT64.
     * @param t1 
     * @param isComplex1 
     * @param t2 
     * @param isComplex2 
     */
    public static bToFloat(t1: DType, isComplex1: boolean, t2: DType, isComplex2: boolean): DType | undefined {
        switch (t1) {
            case DType.LOGIC:
            case DType.INT32:
                return t2 === DType.FLOAT32 ? DType.FLOAT32 : DType.FLOAT64;
            case DType.FLOAT32:
                return t2 === DType.FLOAT64 ? DType.FLOAT64 : DType.FLOAT32;            
            case DType.FLOAT64:
                return DType.FLOAT64;
        }
        return undefined;
    }

    /**
     * Always returns LOGIC.
     * @param t1 
     * @param isComplex1 
     * @param t2 
     * @param isComplex2 
     */
    public static bToLogic(t1: DType, isComplex1: boolean, t2: DType, isComplex2: boolean): DType {
        return DType.LOGIC;
    }

    /**
     * Always returns LOGIC when both inputs are real.
     * Returns undefined when any of the inputs is complex.
     * @param t1 
     * @param isComplex1 
     * @param t2 
     * @param isComplex2 
     */
    public static bToLogicRealOnly(t1: DType, isComplex1: boolean, t2: DType, isComplex2: boolean): DType | undefined {
        return (isComplex1 || isComplex2) ? undefined : DType.LOGIC;
    }

}