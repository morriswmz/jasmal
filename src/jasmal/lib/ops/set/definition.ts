import { Tensor } from '../../tensor';
import { OpInput, Scalar } from '../../commonTypes';

export interface ISetOpProvider {

    /**
     * Obtains the unique elements in the (flattened) input.
     * Note: NaN is not equal to NaN. Therefore no NaN will be removed.
     */
    unique(x: OpInput, outputIndices?: false): Tensor;
    /**
     * Obtains the unique elements in the (flattened) input.
     * Returns a tensor tuple [y, iy, ix] such that y = x[iy] and ix[j]
     * contains the indices in x such that all elements in x[ix[j]] equal to
     * y[j].
     * Note: NaN is not equal to NaN. Therefore no NaN will be removed.
     */
    unique(x: OpInput, outputIndices: true): [Tensor, number[], number[][]];

    ismember(x: Scalar, y: OpInput, outputIndices?: false): boolean;
    ismember(x: Scalar, y: OpInput, outputIndices: true): [boolean, number];
    ismember(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
    ismember(x: OpInput, y: OpInput, outputIndices: true): [Tensor, Tensor];

    union(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
    union(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[], number[]];
    
    intersect(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
    intersect(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[], number[]];

    setdiff(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
    setdiff(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[]];
}
