import { Tensor } from '../../tensor';

export interface IRandomOpProvider {

    /**
     * Specifies the seed for the pseudo random number generator.
     */
    seed(s: number): void;

    /**
     * Obtains a pseudo random number between 0 and 1 (both exclusive).
     */
    rand(): number;
    /**
     * Obtains a tensor of the specified shape filled with pseudo random numbers
     * between 0 and 1 (both exclusive).
     */
    rand(shape: number[]): Tensor;

    /**
     * Samples a pseudo random number from the normal distribution.
     */
    randn(): number;
    /**
     * Obtains a tensor of the specified shape filled with pseudo random numbers
     * sampled from the normal distribution.
     */
    randn(shape: number[]): Tensor;

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
    randi(low: number, high: number, shape: number[]): Tensor;

}