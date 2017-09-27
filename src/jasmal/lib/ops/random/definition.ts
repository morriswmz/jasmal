import { Tensor } from '../../tensor';

export interface IRandomOpProvider {

    /**
     * Specifies the seed for the pseudo random number generator.
     */
    seed(s: number): void;
    /**
     * Retrieves the current seed.
     */
    seed(): number;

    /**
     * Obtains a pseudo random number between 0 and 1 (both exclusive).
     */
    rand(): number;
    /**
     * Obtains a tensor of the specified shape filled with pseudo random numbers
     * between 0 and 1 (both exclusive).
     */
    rand(shape: ArrayLike<number>): Tensor;

    /**
     * Samples a pseudo random number from the normal distribution.
     */
    randn(): number;
    /**
     * Obtains a tensor of the specified shape filled with pseudo random numbers
     * sampled from the normal distribution.
     */
    randn(shape: ArrayLike<number>): Tensor;

    /**
     * Obtains a pseudo random integer ranging from 0 to `high` (both
     * inclusive).
     */
    randi(high: number): number;
    /**
     * Obtains a pseudo random integer ranging from `low` to `high` (both
     * inclusive).
     */
    randi(low: number, high: number): number;
    /**
     * Obtains a tensor of the specified shape filled with pseudo random
     * integers ranging from `low` to `high` (both inclusive).
     */
    randi(low: number, high: number, shape: ArrayLike<number>): Tensor;

    /**
     * Generates a pseudo random real number within (low, high).
     */
    unifrnd(low: number, high: number): number;
    /**
     * Obtains a tensor of the specified shape filled with pseudo random real
     * numbers within (low, high).
     */
    unifrnd(low: number, high: number, shape: ArrayLike<number>): Tensor;

}
