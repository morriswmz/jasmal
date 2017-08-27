import { DataBlock } from '../../../commonTypes';

export class Cholesky {

    public static chol(n: number, a: DataBlock): number {
        let i: number, j: number, k: number;
        let acc: number;
        for (i = 0;i < n;i++) {
            for (j = 0;j <= i;j++) {
                acc = a[i * n + j];
                for (k = 0;k < j;k++) {
                    acc -= a[i * n + k] * a[j * n + k];
                }
                if (i === j) {
                    if (acc <= 0.0) {
                        // not positive definite
                        return n - i - 1;
                    }
                    a[i * n + i] = Math.sqrt(acc);
                } else {
                    a[i * n + j] = acc / a[j * n + j];
                }
            }
        }
        return 0;
    }

    public static cchol(n: number, reA: DataBlock, imA: DataBlock): number {
        let i: number, j: number, k: number;
        let accRe: number, accIm: number;
        for (i = 0;i < n;i++) {
            for (j = 0;j <= i;j++) {
                accRe = reA[i * n + j];
                accIm = imA[i * n + j];
                if (i === j) {
                    if (accIm !== 0) {
                        // non real diagonal
                        return i + 1;
                    }
                    for (k = 0;k < j;k++) {
                        accRe -= reA[j * n + k] * reA[j * n + k] + imA[j * n + k] * imA[j * n + k];
                    }
                    if (accRe <= 0.0) {
                        // not positive definite
                        return i + 1;
                    }
                    reA[i * n + i] = Math.sqrt(accRe);
                    imA[i * n + i] = 0.0;
                } else {
                    for (k = 0;k < j;k++) {
                        accRe -= reA[i * n + k] * reA[j * n + k] + imA[i * n + k] * imA[j * n + k];
                        accIm -= -reA[i * n + k] * imA[j * n + k] + imA[i * n + k] * reA[j * n + k];
                    }
                    reA[i * n + j] = accRe / reA[j * n + j];
                    imA[i * n + j] = accIm / reA[j * n + j];
                }
            }
        }
        return 0;
    }
}
