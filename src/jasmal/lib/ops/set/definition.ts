import { Tensor } from '../../tensor';
import { OpInput, Scalar, NonScalarOpInput } from '../../commonTypes';

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

    /**
     * Checks if x is an element in y.
     */
    ismember(x: Scalar, y: OpInput, outputIndices?: false): boolean;
    /**
     * Checks if x is an element in y. Also returns the index of x in y (-1 if
     * x is not in y).
     */
    ismember(x: Scalar, y: OpInput, outputIndices: true): [boolean, number];
    /**
     * For every element in x, checks if it is an element in y.
     */
    ismember(x: NonScalarOpInput, y: OpInput, outputIndices?: false): Tensor;
    /**
     * For every element in x, checks if it is an element in y. Also returns the
     * indices in y for every element in x (-1 if the element is not in y).
     */
    ismember(x: NonScalarOpInput, y: OpInput, outputIndices: true): [Tensor, Tensor];

    /**
     * Obtains the union of x and y.
     */
    union(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
    /**
     * Obtains the union of x and y.
     */
    union(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[], number[]];
    
    /**
     * Obtains the intersection of x and y.
     */
    intersect(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
    /**
     * Obtains the intersection of x and y.
     */
    intersect(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[], number[]];

    /**
     * Obtains a new set by removing elements that appear in y from x.
     */
    setdiff(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
    /**
     * Obtains a new set by removing elements that appear in y from x.
     */
    setdiff(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[]];
}
