import { IDataOpProvider } from './definition';
import { Tensor } from '../../tensor';
import { NOT_IMPLEMENTED } from '../../constant';
import { OpInput, OpOutput, OpOutputWithIndex, DataBlock } from '../../commonTypes';
import { DataFunction } from './datafun';
import { ComplexNumber } from '../../complexNumber';
import { ShapeHelper } from '../../helper/shapeHelper';
import { DType, OutputDTypeResolver } from '../../dtype';
import { DataHelper } from '../../helper/dataHelper';
import { ICoreOpProvider } from '../core/definition';
import { ReductionOpGenerator } from '../generator/reduction/generator';

export class DataOpProviderFactory {

    public static create(coreOp: ICoreOpProvider, reductionOpGen: ReductionOpGenerator): IDataOpProvider {

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

        function compareWithIndex(a: number, b: number, ia: number, ib: number): number {
            // NaN is treated as the largest number
            if (isNaN(a)) {
                return isNaN(b) ? (ia > ib ? 1 : -1) : 1;
            } else {
                if (isNaN(b)) {
                    return -1;
                }
                if (a > b) {
                    return 1;
                } else if (a < b) {
                    return -1;
                } else {
                    return ia > ib ? 1 : -1;
                }
            }
        }

        function opSort(x: OpInput, dir: 'asc' | 'desc', outputIndices: false): Tensor;
        function opSort(x: OpInput, dir: 'asc' | 'desc', outputIndices: true): [Tensor, number[]];
        function opSort(x: OpInput, dir: 'asc' | 'desc', outputIndices: boolean): Tensor | [Tensor, number[]] {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.hasComplexStorage() && !DataHelper.isArrayAllZeros(X.imagData)) {
                throw new Error('Cannot order complex elements.');
            }
            let n = X.size;
            let Y = Tensor.zeros([n]);
            let dataX = X.realData, dataY = Y.realData;
            let indices = DataHelper.naturalNumbers(n);
            // JavaScript's builtin sort is not stable. Since we want the
            // indices, we can obtain a stable sort by comparing their indices
            // when two elements are equal.
            if (dir === 'asc') {
                indices.sort((ia, ib) => compareWithIndex(dataX[ia], dataX[ib], ia, ib));
            } else {
                indices.sort((ia, ib) => compareWithIndex(dataX[ib], dataX[ia], ia, ib));
            }
            for (let i = 0;i < n;i++) {
                dataY[i] = dataX[indices[i]];
            }
            return outputIndices ? [Y, indices] : Y;
        }

        function opUnique(x: OpInput, outputIndices: false): Tensor;  
        function opUnique(x: OpInput, outputIndices: true): [Tensor, number[], number[][]];        
        function opUnique(x: OpInput, outputIndices: boolean = false): Tensor | [Tensor, number[], number[][]] {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            let n = X.size;
            let indices = DataHelper.naturalNumbers(n);
            let dataRe = X.realData;
            let dataIm: ArrayLike<number>;
            let ix: number[][] = [];
            let iy: number[] = [];
            let i: number, last: number, k: number;
            let curRe: number, curIm: number, newRe: number, newIm: number;
            let nUnique = 1;
            let uniqueRe: number[] = [];
            let uniqueIm: number[] = [];
            let Y: Tensor;
            if (X.hasComplexStorage() && !DataHelper.isArrayAllZeros(X.realData)) {
                dataIm = X.imagData;
                // sort complex numbers
                indices.sort((ia, ib) => {
                    // lexicographic order
                    let a = dataRe[ia], b = dataRe[ib];
                    let reOrder: number;
                    if (isNaN(a)) {
                        reOrder = isNaN(b) ? 0 : 1;
                    } else {
                        if (isNaN(b)) {
                            reOrder = -1;
                        } else {
                            reOrder = a > b ? 1 : (a === b ? 0 : -1);
                        }
                    }
                    return reOrder !== 0 ? reOrder : compareWithIndex(dataIm[ia], dataIm[ib], ia, ib);
                });
                // deduplicate
                last = 0;
                k = indices[0];
                curRe = dataRe[k];
                curIm = dataIm[k];
                for (i = 1;i < n;i++) {
                    newRe = dataRe[indices[i]];
                    newIm = dataIm[indices[i]];
                    if (newRe !== curRe || newIm !== curIm) {
                        // found a different value
                        uniqueRe.push(curRe);
                        uniqueIm.push(curIm);
                        curRe = newRe;
                        curIm = newIm;
                        nUnique++;
                        if (outputIndices) {
                            iy.push(k);
                            k = indices[i];
                            ix.push(indices.slice(last, i));
                            last = i;
                        }
                    }
                }
                // process the last one
                uniqueRe.push(curRe);
                uniqueIm.push(curIm);
                if (outputIndices) {
                    iy.push(k);
                    ix.push(indices.slice(last, i));
                }
                Y = Tensor.fromArray(uniqueRe, uniqueIm, X.dtype);
            } else {
                // sort real numbers
                indices.sort((ia, ib) => compareWithIndex(dataRe[ia], dataRe[ib], ia, ib));
                // deduplicate
                last = 0;
                k = indices[0];
                curRe = dataRe[k];
                for (i = 1;i < n;i++) {
                    newRe = dataRe[indices[i]];
                    if (newRe !== curRe) {
                        // found a different value
                        uniqueRe.push(curRe);
                        curRe = newRe;
                        nUnique++;
                        if (outputIndices) {
                            iy.push(k);
                            k = indices[i];
                            ix.push(indices.slice(last, i));
                            last = i;
                        }
                    }
                }
                // process the last one
                uniqueRe.push(curRe);
                if (outputIndices) {
                    iy.push(k);
                    ix.push(indices.slice(last, i));
                }
                Y = Tensor.fromArray(uniqueRe, [], X.dtype);
            }
            return outputIndices ? [Y, iy, ix] : Y;
        };

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
            if (X.hasComplexStorage() && !DataHelper.isArrayAllZeros(X.imagData)) {
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
            std: opStd,
            var: opVar,
            sort: opSort,
            unique: opUnique,
            hist: opHist
        };

    }

}