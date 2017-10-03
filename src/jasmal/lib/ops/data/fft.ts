import { DataBlock } from '../../commonTypes';
import { SpecialFunction } from '../../math/special';
import { DataHelper } from '../../helper/dataHelper';
import { CMath } from '../../math/cmath';

export class FFT {

    /**
     * In-place fast Fourier transform.
     * @param re (Input/Output) Real part. 
     * @param im (Input/Output) Imaginary part.
     * @param forward (Optional) Set to false to compute the inverse transform.
     *                Default value is true and the forward transform is
     *                computed.
     */
    public static FFT(re: DataBlock, im: DataBlock, forward: boolean = true): void {
        if (SpecialFunction.isPowerOfTwoN(re.length)) {
            FFT.FFTPT(re, im, forward);
        } else {
            FFT.FFTNPT(re, im, forward);
        }
    }

    /**
     * Fast Fourier transform for a vector whose length is a power of two.
     * This implementation is based on Paul Bourke's C implementation:
     * http://paulbourke.net/miscellaneous/dft/
     * @param re (Input/Output) Real part.
     * @param im (Input/Output) Imaginary part.
     * @param forward (Optional) Set to false to compute the inverse transform.
     *                Default value is true and the forward transform is
     *                computed.
     */
    public static FFTPT(re: DataBlock, im: DataBlock, forward: boolean): void {
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

    /**
     * Computes the chirp-z transform.
     *  X_k = \sum_{n = 0}^{N - 1} x[n](A * W^{-k})^{-n}
     * Note: if m is equal to the input vector length. The output vector can
     *       be set to the input vector.
     * @param reX Real part of the input vector.
     * @param imX Imaginary part of the input vector.
     * @param m Length of the transform.
     * @param reW Real part of W. 
     * @param imW Imaginary part of W.
     * @param reA Real part of output vector.
     * @param imA Imaginary part of output vector.
     */
    public static CZT(reX: ArrayLike<number>, imX: ArrayLike<number>, m: number,
                      reW: number, imW: number, reA: number, imA: number,
                      reO: DataBlock, imO: DataBlock): void
    {
        let n = reX.length;
        let nn = n + Math.max(m, n) - 1;
        let n2 = SpecialFunction.nextPowerOfTwo(m + n - 1);
        let i: number;
        // Using the property that kn = 0.5(k^2 + n^2 - (k - n)^2),
        // we can evalute CZT via fast convolution.
        // Compute the chirp
        let reC = DataHelper.allocateFloat64Array(nn);
        let imC = DataHelper.allocateFloat64Array(nn);
        let phase = Math.atan2(imW, reW);
        let amp = CMath.length2(reW, imW);
        let s: number;
        let ii: number;
        for (i = 1 - n;i < Math.max(m, n);i++) {
            ii = i * i * 0.5;
            s = Math.pow(amp, ii);
            reC[i + n - 1] = s * Math.cos(phase * ii);
            imC[i + n - 1] = s * Math.sin(phase * ii);
        }
        // Inverse of the chirp
        let reIC = DataHelper.allocateFloat64Array(n2);
        let imIC = DataHelper.allocateFloat64Array(n2);
        for (i = 0;i < m + n - 1;i++) {
            [reIC[i], imIC[i]] = CMath.cdivRC(1.0, reC[i], imC[i]);
        }
        // Prepare for convolution
        let reZ = DataHelper.allocateFloat64Array(n2);
        let imZ = DataHelper.allocateFloat64Array(n2);
        phase = Math.atan2(imA, reA);
        amp = CMath.length2(reA, imA);
        // A^i
        if (reA === 1 && imA === 0) {
            for (i = 0;i < n;i++) {
                reZ[i] = 1;
                imZ[i] = 0;
            }
        } else {
            for (i = 0;i < n;i++) {
                s = Math.pow(amp, i);
                reZ[i] = s * Math.cos(phase * i);
                imZ[i] = s * Math.sin(phase * i);
            }
        }
        for (i = 0;i < n;i++) {
            // * chirp[i]
            s = reZ[i];
            reZ[i] = s * reC[i + n - 1] - imZ[i] * imC[i + n - 1];
            imZ[i] = s * imC[i + n - 1] + imZ[i] * reC[i + n - 1];
            // * x[i]
            s = reZ[i];
            reZ[i] = s * reX[i] - imZ[i] * imX[i];
            imZ[i] = s * imX[i] + imZ[i] * reX[i];
        }
        // Fast convolution via FFT
        FFT.FFTPT(reZ, imZ, true);
        FFT.FFTPT(reIC, imIC, true);
        for (i = 0;i < n2;i++) {
            s = reZ[i];
            reZ[i] = s * reIC[i] - imZ[i] * imIC[i];
            imZ[i] = s * imIC[i] + imZ[i] * reIC[i];
        }
        FFT.FFTPT(reZ, imZ, false);
        // Multiply by the chirp and store the results back
        for (i = 0;i < m;i++) {
            reO[i] = reZ[i + n - 1] * reC[i + n - 1] - imZ[i + n - 1] * imC[i + n - 1];
            imO[i] = reZ[i + n - 1] * imC[i + n - 1] + imZ[i + n - 1] * reC[i + n - 1];
        }
    }

    /**
     * Fast Fourier transform for a vector whose length is not a power of two.
     * We simply use chirp-z transform here. It is possible to have more efficient
     * implementations.
     * @param re (Input/Output) Real part.
     * @param im (Input/Output) Imaginary part.
     * @param forward (Optional) Set to false to compute the inverse transform.
     *                Default value is true and the forward transform is
     *                computed.
     */
    public static FFTNPT(re: DataBlock, im: DataBlock, forward: boolean): void {
        let n = re.length;
        let reW: number, imW: number;
        reW = Math.cos(2 * Math.PI / n);
        if (forward) {
            imW = -Math.sin(2 * Math.PI / n);
        } else {
            imW = Math.sin(2 * Math.PI / n);
        }
        FFT.CZT(re, im, n, reW, imW, 1, 0, re, im);
        if (!forward) {
            // Inverse scaling
            let nInv = 1.0 / n;
            for (let i = 0;i < n;i++) {
                re[i] *= nInv;
                im[i] *= nInv;
            }
        }
    }

}
