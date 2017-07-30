import { OpOutput, OpInput } from '../../commonTypes';

export interface ILogicComparisonOpProvider {

    eq(x: OpInput, y: OpInput): OpOutput;

    neq(x: OpInput, y: OpInput): OpOutput;

    gt(x: OpInput, y: OpInput): OpOutput;

    ge(x: OpInput, y: OpInput): OpOutput;

    lt(x: OpInput, y: OpInput): OpOutput;

    le(x: OpInput, y: OpInput): OpOutput;

    and(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    or(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    xor(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    not(x: OpInput, inPlace?: boolean): OpOutput;

    all(x: OpInput): boolean;

    any(x: OpInput): boolean;            

}