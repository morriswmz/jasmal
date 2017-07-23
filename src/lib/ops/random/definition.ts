import { Tensor } from '../../tensor';

export interface IRandomOpProvider {

    seed(s: number): void;

    rand(): number;
    rand(shape: number[]): Tensor;

    randn(): number;
    randn(shape: number[]): Tensor;

    randi(high: number): number;
    randi(low: number, high: number): number;
    randi(low: number, high: number, shape: number[]): Tensor;

}