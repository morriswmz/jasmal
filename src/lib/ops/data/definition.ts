import { OpInput, OpOutputWithIndex, OpOutput } from '../../commonTypes';
import { Tensor } from '../../tensor';

// TODO: add sortRows, unique, hist
export interface IDataOpProvider {

    min(x: OpInput, axis?: number, keepDims?: boolean): OpOutputWithIndex;

    max(x: OpInput, axis?: number, keepDims?: boolean): OpOutputWithIndex;

    sum(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    prod(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    cumsum(x: OpInput, axis?: number): Tensor;

    mean(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    median(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    var(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    std(x: OpInput, axis?: number, keepDims?: boolean): OpOutput;

    sort(x: OpInput, dir: 'asc' | 'desc', outputIndices: false): Tensor;
    sort(x: OpInput, dir: 'asc' | 'desc', outputIndices: true): [Tensor, number[]];

}