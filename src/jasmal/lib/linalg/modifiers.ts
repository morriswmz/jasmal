/**
 * Specifies whether the matrix should be (Hermitian) transposed prior to
 * further operations.
 */
export const enum MatrixModifier {
    /**
     * Uses the original matrix.
     */
    None = 0,
    /**
     * Uses the transposed version of the original matrix.
     */
    Transposed = 1,
    /**
     * Uses the Hermitian transposed version of the original matrix.
     */
    Hermitian = 2
}
