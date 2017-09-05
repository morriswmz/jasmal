import { IDataOpProvider } from './definition';
import { Tensor } from '../../tensor';
import { OpInput, DataBlock } from '../../commonTypes';
import { DataFunction } from './datafun';
import { OutputDTypeResolver, DType } from '../../dtype';
import { DataHelper } from '../../helper/dataHelper';
import { ICoreOpProvider } from '../core/definition';
import { ReductionOpGenerator } from '../generator';
import { ComparisonHelper } from '../../helper/comparisonHelper';
import { SpecialFunction } from '../../math/special';
import { FFT } from './fft';
import { IArithmeticOpProvider } from '../arithmetic/definition';
import { IMatrixOpProvider, MatrixModifier } from '../matrix/definition';
import { IMathOpProvider } from '../math/definition';

export class DataOpProviderFactory {

    public static create(coreOp: ICoreOpProvider, arithOp: IArithmeticOpProvider, mathOp: IMathOpProvider,
                         matOp: IMatrixOpProvider, reductionOpGen: ReductionOpGenerator): IDataOpProvider {

        const opMin = reductionOpGen.makeRealOnlyOpWithIndexOutput(
            DataFunction.min, { outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat });

        const opMax = reductionOpGen.makeRealOnlyOpWithIndexOutput(
            DataFunction.max, { outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat });

        const opSum = reductionOpGen.makeOp(
            DataFunction.sum, (reX, imX, offset, stride, n) => {
                return [DataFunction.sum(reX, offset, stride, n),
                        DataFunction.sum(imX, offset, stride, n)]
            }, true, { outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat });

        const opProd = reductionOpGen.makeOp(
            DataFunction.prod, DataFunction.cprod, true,
            { outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat });

        const opMean = reductionOpGen.makeOp(
            (reX, offset, stride, n) => {
                return DataFunction.sum(reX, offset, stride, n) / n;
            }, (reX, imX, offset, stride, n) => {
                return [DataFunction.sum(reX, offset, stride, n) / n,
                        DataFunction.sum(imX, offset, stride, n) / n];
            }, true, { outputDTypeResolver: OutputDTypeResolver.uToFloat });

        const opMedian = reductionOpGen.makeRealOnlyOp(
            DataFunction.median, { outputDTypeResolver: OutputDTypeResolver.uToFloat });
        
        const opMode = reductionOpGen.makeRealOnlyOp(
            DataFunction.mode, { outputDTypeResolver: OutputDTypeResolver.uNoChange });

        const opVar = reductionOpGen.makeOp(
            DataFunction.var, DataFunction.cvar, false,
            { outputDTypeResolver: OutputDTypeResolver.uToFloat });

        const opStd = reductionOpGen.makeOp(
            (reX, offset, stride, n) => {
                return Math.sqrt(DataFunction.var(reX, offset, stride, n));
            }, (reX, imX, offset, stride, n) => {
                return Math.sqrt(DataFunction.cvar(reX, imX, offset, stride, n));
            }, false, {outputDTypeResolver: OutputDTypeResolver.uToFloat });

        const opCumsum = (x: OpInput, axis: number = -1): Tensor => {
            let X = x instanceof Tensor ? x.copy(true) : Tensor.toTensor(x);
            // cumsum can be done in place
            if (axis < 0 || (axis === 0 && X.ndim === 1)) {
                let re = X.realData;
                for (let i = 1;i < re.length;i++) {
                    re[i] += re[i - 1];
                }
                if (X.hasComplexStorage()) {
                    let im = X.imagData;
                    for (let i = 1;i < im.length;i++) {
                        im[i] += im[i - 1];
                    }
                }
                X.reshape([-1]);
            } else {
                // check axis
                if (axis >= X.ndim) {
                    throw new Error(`Invalid axis number ${axis}.`);
                }
                let shape = X.shape;
                let n = shape[axis];
                if (n !== 1) {
                    // no need to do any thing if n = 1
                    let strides = X.strides;
                    let stride = strides[axis];
                    let maxLevel = X.ndim - 1;
                    let doCumsum = (data: DataBlock, level: number, offset: number): void => {
                        if (level === maxLevel) {
                            for (let i = level === axis ? 1 : 0;i < shape[level];i++) {
                                data[offset + i] += data[offset + i - stride];
                            }
                        } else {
                            let i;
                            if (level === axis) {
                                i = 1;
                                offset += strides[level];
                            } else {
                                i = 0;
                            }
                            for (;i < shape[level];i++) {
                                doCumsum(data, level + 1, offset);
                                offset += strides[level];
                            }
                        }
                    };
                    doCumsum(X.realData, 0, 0);
                    if (X.hasComplexStorage()) {
                        doCumsum(X.imagData, 0, 0);
                    }
                }
            }
            return X;
        };

        const opCov = (x: OpInput, y: OpInput = x, samplesInColumns: boolean = true): Tensor => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.ndim === 1) {
                X = coreOp.prependAxis(X);
            } else if (X.ndim > 2) {
                throw new Error('Input must be either 1D or 2D.');
            }
            let axis = samplesInColumns ? 1 : 0;
            let n = X.shape[axis];
            let mX: Tensor = <Tensor>opMean(X, axis, true);
            let xM = arithOp.sub(X, mX);
            if (y === x) {
                // covariance matrix
                if (samplesInColumns) {
                    return <Tensor>arithOp.div(matOp.matmul(xM, xM, MatrixModifier.Hermitian), n - 1);
                } else {
                    xM = <Tensor>mathOp.conj(xM, true);
                    return <Tensor>arithOp.div(matOp.matmul(matOp.hermitian(xM), xM), n - 1);
                }
            } else {
                let Y = y instanceof Tensor ? y : Tensor.toTensor(y);
                if (Y.ndim === 1) {
                    Y = coreOp.prependAxis(X);
                } else if (Y.ndim > 2) {
                    throw new Error('Input must be either 1D or 2D.');
                }
                if (Y.shape[axis] !== n) {
                    throw new Error('x and y must share the same number of samples.');
                }
                // cross covariance matrix
                let mY: Tensor = <Tensor>opMean(Y, axis, true);
                let yM = arithOp.sub(Y, mY);
                if (samplesInColumns) {
                    return <Tensor>arithOp.div(matOp.matmul(xM, yM, MatrixModifier.Hermitian), n - 1);
                } else {
                    yM = <Tensor>mathOp.conj(yM, true);
                    return <Tensor>arithOp.div(matOp.matmul(matOp.hermitian(xM), yM), n - 1);
                }
            }
        };

        const opCorrcoef = (x: OpInput, y: OpInput = x, samplesInColumns: boolean = true): Tensor => {
            let auto = x === y;
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            let Y = auto ? X : (y instanceof Tensor ? y : Tensor.toTensor(y));
            let C = opCov(X, Y, samplesInColumns);
            if (auto) {
                // we can extra the standard deviation from the diagonals directly
                let v = matOp.diag(C).trimImaginaryPart();
                v = <Tensor>mathOp.sqrt(v, true);
                arithOp.div(C, v, true);
                arithOp.div(C, coreOp.appendAxis(v), true);
            } else {
                let axis = samplesInColumns ? 1 : 0;
                let stdX = opStd(X, axis);
                let stdY = opStd(Y, axis);
                arithOp.div(C, coreOp.appendAxis(stdX), true);
                arithOp.div(C, coreOp.prependAxis(stdY), true);
            }
            return C;
        };

        const opFFTFB = (x: OpInput, forward: boolean = true, axis: number = -1): Tensor => {
            let X = x instanceof Tensor ? x.asType(DType.FLOAT64, true) : Tensor.toTensor(x);
            X.ensureComplexStorage();
            if (axis < 0 || (axis === 0 && X.ndim === 1)) {
                if (SpecialFunction.isPowerOfTwoN(X.size)) {
                    FFT.FFT(X.realData, X.imagData, forward);
                } else {
                    FFT.FFTNoPT(X.realData, X.imagData, forward);
                }
            } else {
                if (axis >= X.ndim) {
                    throw new Error(`Invalid axis number ${axis}.`);
                }
                let shapeX = X.shape;
                let n = shapeX[axis];
                let FFTFunc = SpecialFunction.isPowerOfTwoN(n) ? FFT.FFT : FFT.FFTNoPT;
                let tmpReArr = DataHelper.allocateFloat64Array(n);
                let tmpImArr = DataHelper.allocateFloat64Array(n);
                let strides = X.strides;
                let strideAtAxis = strides[axis];
                let maxLevel = X.ndim - 1;
                let doFFT = (re: DataBlock, im: DataBlock, level: number, offset: number): void => {
                    if (level === maxLevel) {
                        for (let i = 0;i < shapeX[level];i++) {
                            // copy data to tmp array
                            for (let k = 0;k < n;k++) {
                                tmpReArr[k] = re[strideAtAxis * k + offset];
                                tmpImArr[k] = im[strideAtAxis * k + offset];
                            }
                            // do FFT
                            FFTFunc(tmpReArr, tmpImArr, forward);
                            // copy back
                            for (let k = 0;k < n;k++) {
                                re[strideAtAxis * k + offset] = tmpReArr[k];
                                im[strideAtAxis * k + offset] = tmpImArr[k];
                            }
                            offset++;
                        }
                    } else {
                        // recursive calling
                        let maxI = level === axis ? 1 : shapeX[level];
                        for (let i = 0;i < maxI;i++) {
                            doFFT(re, im, level + 1, offset);
                            offset += strides[level];
                        }
                    }
                }
                doFFT(X.realData, X.imagData, 0, 0);
            }
            return X;
        };

        const opFFT = (x: OpInput, axis: number = -1): Tensor => opFFTFB(x, true, axis);
        const opIFFT = (x: OpInput, axis: number = -1): Tensor => opFFTFB(x, false, axis);

        function opSort(x: OpInput, dir: 'asc' | 'desc', outputIndices: false): Tensor;
        function opSort(x: OpInput, dir: 'asc' | 'desc', outputIndices: true): [Tensor, number[]];
        function opSort(x: OpInput, dir: 'asc' | 'desc', outputIndices: boolean): Tensor | [Tensor, number[]] {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.hasNonZeroComplexStorage()) {
                throw new Error('Cannot order complex elements.');
            }
            let n = X.size;
            let Y = Tensor.zeros([n]);
            let dataX = X.realData, dataY = Y.realData;
            let indices = DataHelper.naturalNumbers(n);
            // JavaScript's builtin sort is not stable. Since we want the
            // indices, we can obtain a stable sort by comparing their indices
            // when two elements are equal.
            let comparator = ComparisonHelper.compareNumberWithIndexAsc;
            if (dir === 'asc') {
                indices.sort((ia, ib) => comparator(dataX[ia], dataX[ib], ia, ib));
            } else {
                indices.sort((ia, ib) => comparator(dataX[ib], dataX[ia], ia, ib));
            }
            for (let i = 0;i < n;i++) {
                dataY[i] = dataX[indices[i]];
            }
            return outputIndices ? [Y, indices] : Y;
        }

        function opSortRows(x: OpInput, dir: 'asc' | 'desc', outputIndices: false): Tensor;
        function opSortRows(x: OpInput, dir: 'asc' | 'desc', outputIndices: true): [Tensor, number[]];
        function opSortRows(x: OpInput, dir: 'asc' | 'desc', outputIndices: boolean): Tensor | [Tensor, number[]] {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.ndim !== 2) {
                throw new Error('Matrix expected.');
            }
            if (X.hasNonZeroComplexStorage()) {
                throw new Error('Cannot order complex elements.');
            }
            let dataX = X.realData;
            let [m, n] = X.shape;
            let indices = DataHelper.naturalNumbers(m);
            let comparator = dir === 'asc' ? ComparisonHelper.compareNumberAsc : ComparisonHelper.compareNumberDesc;
            indices.sort((ia, ib) => {
                for (let j = 0;j < n;j++) {
                    let cur = comparator(dataX[ia * n + j], dataX[ib * n + j]);
                    if (cur !== 0) {
                        return cur;
                    }
                }
                // stable sort trick
                return (ia > ib) ? 1 : -1;
            });
            let Y = <Tensor>X.get(indices, ':', true);
            return outputIndices ? [Y, indices] : Y;
        }

        function findBinIndex(x: number, edges: ArrayLike<number>): number {
            if (edges.length < 2) {
                throw new Error('Number of edges must be at least 2.');
            }
            if (x >= edges[edges.length - 1]) {
                return edges.length - 2;
            }
            // binary search
            let l = 0, u = edges.length - 1;
            let i: number;
            while (u - l > 1) {
                i = (l + u) >>> 1;
                if (x < edges[i]) {
                    u = i;
                } else if (x > edges[i]) {
                    l = i;
                } else {
                    return i;
                }
            }
            return l;
        }

        const opHist = (x: OpInput, nBins: number = 10): [Tensor, Tensor] => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.hasNonZeroComplexStorage()) {
                throw new Error('Input must be real.');
            }
            if (nBins <= 0 || Math.floor(nBins) !== nBins) {
                throw new Error('Number of bins must be a positive integer.');
            }
            let i: number;
            let data = X.realData;
            // determine the range, NaNs are ignored
            // number of negative infinities will be added to the first bin
            // number of positive infinities will be added to the last bin
            let lb = Infinity;
            let ub = -Infinity;
            let nNegInf = 0;
            let nPosInf = 0;
            let nNaN = 0;
            for (i = 0;i < data.length;i++) {
                if (isNaN(data[i])) {
                    nNaN++;
                    continue;
                }
                if (isFinite(data[i])) {
                    if (data[i] < lb) {
                        lb = data[i];
                    }
                    if (data[i] > ub) {
                        ub = data[i];
                    }
                } else {
                    if (data[i] < 0) {
                        nNegInf++;
                    } else {
                        nPosInf++;
                    }
                }
            }
            if (lb === Infinity && ub === -Infinity) {
                throw new Error('None of the element is finite.');
            }
            let H = Tensor.zeros([nBins]);
            let h = H.realData;
            let E: Tensor;
            let edges: DataBlock;
            if (lb === ub) {
                // special case, only one value
                E = Tensor.zeros([nBins + 1]);
                edges = E.realData;
                // edges[idxM] = edgeM so that lb fall into
                // [edges[idxM], edges[idxM] + 1)
                let idxM = (nBins - 1) >>> 1;
                let edgeM = Math.floor(lb); 
                for (i = 0;i <= nBins;i++) {
                    edges[i] = edgeM - idxM + i;
                }
                h[idxM] = data.length - nNegInf - nPosInf - nNaN;
            } else {
                // general case
                E = coreOp.linspace(lb, ub, nBins + 1);
                edges = E.realData;
                for (i = 0;i < data.length;i++) {
                    if (isFinite(data[i])) {
                        h[findBinIndex(data[i], edges)]++;
                    }
                }
            }
            // count infinities
            h[0] += nNegInf;
            h[nBins - 1] += nPosInf;
            return [H, E];
        };

        return {
            min: opMin,
            max: opMax,
            sum: opSum,
            prod: opProd,
            cumsum: opCumsum,
            mean: opMean,
            median: opMedian,
            mode: opMode,
            std: opStd,
            var: opVar,
            cov: opCov,
            corrcoef: opCorrcoef,
            sort: opSort,
            sortRows: opSortRows,
            hist: opHist,
            fft: opFFT,
            ifft: opIFFT
        };

    }

}
