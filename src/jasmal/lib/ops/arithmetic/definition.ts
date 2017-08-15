import { OpInput, OpOutput } from '../../commonTypes';

export interface IArithmeticOpProvider {

    /**
     * Performs element-wise addition between two compatible inputs.
     * @param x
     * @param y
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    add(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;
    
    /**
     * Performs element-wise subtraction between two compatible inputs.
     * @param x
     * @param y
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    sub(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Performs element-wise negation.
     * @param x
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    neg(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Performs element-wise multiplication between two compatible inputs.
     * @param x
     * @param y
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    mul(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Performs element-wise division between two compatible inputs.
     * @param x
     * @param y
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    div(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Evaluates the reciprocal of each element in the input.
     * @param x
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    reciprocal(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Evaluates remainder after division (using JavaScript's % operator)
     * between two compatible inputs.
     * Note: this is **not** modulo.
     * @param x
     * @param inPlace (Optional) If set to true, the operation will be performed
     *                in place (i.e., the results will be stored in `x` instead
     *                of a new tensor. `x` must be a tensor with compatible
     *                shape and DType. Default value is false. 
     */
    rem(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

}