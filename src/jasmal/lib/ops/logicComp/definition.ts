import { OpInput, RealOpOutput, RealOpInput } from '../../commonTypes';

export interface ILogicComparisonOpProvider {

    /**
     * Evaluates `x === y` element-wise for two compatible inputs x and y.
     */
    eq(x: OpInput, y: OpInput): RealOpOutput;

    /**
     * Evaluates `x !== y` element-wise for two compatible inputs x and y.
     */
    neq(x: OpInput, y: OpInput): RealOpOutput;

    /**
     * Evaluates `x > y` element-wise for two compatible inputs x and y.
     */
    gt(x: RealOpInput, y: RealOpInput): RealOpOutput;

    /**
     * Evaluates `x >= y` element-wise for two compatible inputs x and y.
     */
    ge(x: RealOpInput, y: RealOpInput): RealOpOutput;

    /**
     * Evaluates `x < y` element-wise for two compatible inputs x and y.
     */
    lt(x: RealOpInput, y: RealOpInput): RealOpOutput;

    /**
     * Evaluates `x <= y` element-wise for two compatible inputs x and y.
     */
    le(x: RealOpInput, y: RealOpInput): RealOpOutput;

    /**
     * Evaluates `(x !== 0) && (y !== 0)` element-wise for two compatible inputs x and y.
     */
    and(x: RealOpInput, y: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Evaluates `(x !== 0) || (y !== 0)` element-wise for two compatible inputs x and y.
     */
    or(x: RealOpInput, y: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Evaluates `(x !== 0) ^ (y !== 0)` element-wise for two compatible inputs x and y.
     */
    xor(x: RealOpInput, y: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Evaluates `!(x !== 0)` element-wise.
     */
    not(x: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Returns true only if all the elements in the input are non-zero.
     */
    all(x: OpInput): boolean;

    /**
     * Returns true if any of the elements in the input is non-zero.
     */
    any(x: OpInput): boolean;

}
