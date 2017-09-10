import { RealOpInput, RealOpOutput } from '../../commonTypes';

export interface IBinaryOpProvider {
    
    /**
     * Computes bitwise AND between two compatible inputs using JavaScript's
     * `&` operator, following JavaScript's conversion rules for non integers.
     * @param x
     * @param y
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false.
     */
    bitwiseAnd(x: RealOpInput, y: RealOpInput, inPlace?: boolean): RealOpOutput;
    
    /**
     * Computes bitwise OR between two compatible inputs using JavaScript's
     * `|` operator, following JavaScript's conversion rules for non integers.
     * @param x
     * @param y
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    bitwiseOr(x: RealOpInput, y: RealOpInput, inPlace?: boolean): RealOpOutput;
    
    /**
     * Computes bitwise XOR between two compatible inputs using JavaScript's
     * `^` operator, following JavaScript's conversion rules for non integers.
     * @param x
     * @param y
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    bitwiseXor(x: RealOpInput, y: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Computes bitwise NOT for each element in the input using JavaScript's
     * `~` operator, following JavaScript's conversion rules for non integers.
     * @param x
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    bitwiseNot(x: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Shifts the bits of x to the left by the amount specified by y using
     * JavaScript's `<<` operator, following JavaScript's conversion rules for
     * non integers.
     * @param x
     * @param y
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    leftShift(x: RealOpInput, y: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Shifts the bits of x to the right by the amount specified by y with the
     * sign bit propagated, using JavaScript's `>>` operator, following
     * JavaScript's conversion rules for non integers.
     * @param x
     * @param y
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    rightShiftSP(x: RealOpInput, y: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Shifts the bits of x to the right by the amount specified by y with zero
     * filling, using JavaScript's `>>>` operator, following JavaScript's
     * conversion rules for non integers.
     * @param x
     * @param y
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    rightShiftZF(x: RealOpInput, y: RealOpInput, inPlace?: boolean): RealOpOutput;

}
