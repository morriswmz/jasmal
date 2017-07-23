import { OpInput, OpOutput, Scalar } from '../../commonTypes';
import { DType } from '../../dtype';
import { Tensor } from '../../tensor';

export const enum MatrixModifier {
    None = 0,
    Transposed = 1,
    Hermitian = 2
}

export interface IMatrixOpProvider {

    isSymmetric(x: OpInput, skew?: boolean): boolean;

    isHermitian(x: OpInput, skew?: boolean): boolean;

    eye(m: number, n?: number, dtype?: DType): Tensor;

    hilb(n: number): Tensor;

    diag(x: OpInput): Tensor;

    matmul(x: OpInput, y: OpInput, yModifier?: MatrixModifier): OpOutput;

    kron(x: OpInput, y: OpInput): Tensor;

    transpose(x: OpInput): Tensor;

    hermitian(x: OpInput): Tensor;

    trace(x: OpInput): Scalar;

    inv(x: OpInput): Tensor;

    det(x: OpInput): Scalar;

    norm(x: OpInput, p: number | 'fro'): number;

    lu(x: OpInput, compact: true): [Tensor, number[]];
    lu(x: OpInput, compact: false): [Tensor, Tensor, Tensor];

    svd(x: OpInput): [Tensor, Tensor, Tensor];

    rank(x: OpInput): number;

    eig(x: OpInput): [Tensor, Tensor];

}