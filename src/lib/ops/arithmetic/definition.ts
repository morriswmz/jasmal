import { OpInput, OpOutput } from '../../commonTypes';

export interface IArithmeticOpProvider {

    add(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;
    
    sub(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    neg(x: OpInput, inPlace?: boolean): OpOutput;

    mul(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    div(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    reciprocal(x: OpInput, inPlace?: boolean): OpOutput;

    rem(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

}