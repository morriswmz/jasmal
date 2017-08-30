import { DataBlock } from '../../commonTypes';
import { SpecialFunction } from '../../math/special';

export class FFT {

    /**
     * Fast Fourier transform for a vector whose length is a power of two.
     * This implementation is based on Paul Bourke's C implementation:
     * http://paulbourke.net/miscellaneous/dft/
     * @param re (Input/Output) Real part.
     * @param im (Input/Output) Imaginary part.
     */
    public static FFT(re: DataBlock, im: DataBlock, forward: boolean = true): void {
        let n = re.length;
        if (im.length !== n) {
            throw new Error('Real part and imaginary part must have the same length.');
        }
        if (!SpecialFunction.isPowerOfTwoN(n)) {
            throw new Error('Length must be a power of 2.');
        }
        let i: number, i1: number, i2: number, j: number, k: number;
        let tRe: number, tIm: number;
        // Bit-reversal
        i2 = n >> 1;
        j = 0; // index obtained by reversing the bits in i
        for (i = 0;i < n - 1;i++) {
           if (i < j) {
              tRe = re[i];
              tIm = im[i];
              re[i] = re[j];
              im[i] = im[j];
              re[j] = tRe;
              im[j] = tIm;
           }
           // updated j
           k = i2;
           while (k <= j) {
              j -= k;
              k >>= 1;
           }
           j += k;
        }     
        // FFT
        let m = 0;
        k = 1;
        while (!(n & k)) {
            m++;
            k <<= 1;
        }
        let l: number, l1: number, uRe: number, uIm: number, tmp: number;
        let c = -1.0; 
        let s = 0.0;
        let l2 = 1;
        for (l = 0;l < m;l++) {
           l1 = l2;
           l2 <<= 1;
           uRe = 1.0; 
           uIm = 0.0;
           for (j = 0;j < l1;j++) {
              for (i = j;i < n;i += l2) {
                 i1 = i + l1;
                 tRe = uRe * re[i1] - uIm * im[i1];
                 tIm = uRe * im[i1] + uIm * re[i1];
                 re[i1] = re[i] - tRe; 
                 im[i1] = im[i] - tIm;
                 re[i] += tRe;
                 im[i] += tIm;
              }
              tmp =  uRe * c - uIm * s;
              uIm = uRe * s + uIm * c;
              uRe = tmp;
           }
           // sin(x/2) = \pm sqrt((1-cos(x))/2)
           // cos(x/2) = \pm sqrt((1+cos(x))/2)
           s = forward ? -Math.sqrt((1.0 - c) / 2.0) : Math.sqrt((1.0 - c) / 2.0);
           c = Math.sqrt((1.0 + c) / 2.0);
        }
        // Inverse scaling
        if (!forward) {
            let nInv = 1.0 / n;
            for (i = 0;i < n;i++) {
                re[i] *= nInv;
                im[i] *= nInv;
            }
        }
    }

    public static FFTNoPT(_re: DataBlock, _im: DataBlock, _forward: boolean = true): void {
        throw new Error('Not implemented.');
    }

}
