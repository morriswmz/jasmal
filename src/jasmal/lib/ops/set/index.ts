import { ISetOpProvider } from './definition';
import { OpInput, Scalar, NonScalarOpInput, DataBlock, OpInputInfo } from '../../commonTypes';
import { Tensor } from '../../core/tensor';
import { DataHelper } from '../../helper/dataHelper';
import { ComparisonHelper } from '../../helper/comparisonHelper';
import { ICoreOpProvider } from '../core/definition';
import { ILogicComparisonOpProvider } from '../logicComp/definition';
import { DTypeHelper, DType } from '../../core/dtype';
import { IJasmalModuleFactory, JasmalOptions } from '../../jasmal';

export class SetOpProviderFactory implements IJasmalModuleFactory<ISetOpProvider> {

    constructor(private _coreOp: ICoreOpProvider, private _logicCompOp: ILogicComparisonOpProvider) {
    }

    public create(_options: JasmalOptions): ISetOpProvider {

        const coreOp = this._coreOp;
        const logicCompOp = this._logicCompOp;

        function opUniqueInternal(infoX: OpInputInfo, outputIndices: false): [Tensor, number[]];
        function opUniqueInternal(infoX: OpInputInfo, outputIndices: true): [Tensor, number[], number[][]];
        function opUniqueInternal(infoX: OpInputInfo, outputIndices: boolean): [Tensor, number[]] | [Tensor, number[], number[][]] {
            // empty input handling
            if (!infoX.isInputScalar && infoX.reArr.length === 0) {
                let empty = Tensor.zeros([0], infoX.originalDType);
                return outputIndices ? [empty, [], []] : [empty, []];
            }
            let dataRe: ArrayLike<number> = infoX.isInputScalar ? [infoX.re] : infoX.reArr;
            let dataIm: ArrayLike<number>;
            let n = dataRe.length;
            let indices = DataHelper.naturalNumbers(n);
            let ix: number[][] = [];
            let iy: number[] = [];
            let i: number, last: number, k: number;
            let curRe: number, curIm: number, newRe: number, newIm: number;
            let uniqueRe: DataBlock;
            let uniqueIm: DataBlock;
            let Y: Tensor;
            if (infoX.isComplex) {
                dataIm = infoX.isInputScalar ? [infoX.im] : infoX.imArr;
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
                    return reOrder !== 0 ? reOrder : ComparisonHelper.compareNumberWithIndexAsc(dataIm[ia], dataIm[ib], ia, ib);
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
                        curRe = newRe;
                        curIm = newIm;
                        iy.push(k);
                        k = indices[i];
                        if (outputIndices) {
                            ix.push(indices.slice(last, i));
                            last = i;
                        }
                    }
                }
                // process the last one
                iy.push(k);
                if (outputIndices) {
                    ix.push(indices.slice(last, i));
                }
                Y = Tensor.zeros([iy.length], infoX.originalDType).ensureComplexStorage();
                // copy data
                uniqueRe = Y.realData;
                uniqueIm = Y.imagData;
                for (i = 0;i < uniqueRe.length;i++) {
                    uniqueRe[i] = dataRe[iy[i]];
                    uniqueIm[i] = dataIm[iy[i]];
                }
            } else {
                // sort real numbers
                indices.sort((ia, ib) => ComparisonHelper.compareNumberWithIndexAsc(dataRe[ia], dataRe[ib], ia, ib));
                // deduplicate
                last = 0;
                k = indices[0];
                curRe = dataRe[k];
                for (i = 1;i < n;i++) {
                    newRe = dataRe[indices[i]];
                    if (newRe !== curRe) {
                        // found a different value
                        curRe = newRe;
                        iy.push(k);
                        k = indices[i];
                        if (outputIndices) {
                            ix.push(indices.slice(last, i));
                            last = i;
                        }
                    }
                }
                // process the last one
                iy.push(k);
                if (outputIndices) {
                    ix.push(indices.slice(last, i));
                }
                // copy data
                Y = Tensor.zeros([iy.length], infoX.originalDType);
                // copy data
                uniqueRe = Y.realData;
                for (i = 0;i < uniqueRe.length;i++) {
                    uniqueRe[i] = dataRe[iy[i]];
                }
            }
            return outputIndices ? [Y, iy, ix] : [Y, iy];
        };

        function opUnique(x: OpInput, outputIndices?: false): Tensor;  
        function opUnique(x: OpInput, outputIndices: true): [Tensor, number[], number[][]];        
        function opUnique(x: OpInput, outputIndices: boolean = false): Tensor | [Tensor, number[], number[][]] {
            let infoX = Tensor.analyzeOpInput(x);
            return outputIndices ? opUniqueInternal(infoX, true) : opUniqueInternal(infoX, false)[0];
        };

        function opIsIn(x: Scalar, y: OpInput, outputIndices?: false): boolean;
        function opIsIn(x: Scalar, y: OpInput, outputIndices: true): [boolean, number];
        function opIsIn(x: NonScalarOpInput, y: OpInput, outputIndices?: false): Tensor;
        function opIsIn(x: NonScalarOpInput, y: OpInput, outputIndices: true): [Tensor, Tensor];
        function opIsIn(x: OpInput, y: OpInput, outputIndices: boolean = false):
            boolean | Tensor | [boolean, number] | [Tensor, Tensor]
        {
            let infoX = Tensor.analyzeOpInput(x);
            let infoY = Tensor.analyzeOpInput(y);
            if (infoX.isInputScalar) {
                // Special treatment when x is scalar.
                if (infoY.isInputScalar) {
                    if (infoX.re === infoY.re && infoX.im === infoY.im) {
                        return outputIndices ? [true, 0] : true;
                    } else {
                        return outputIndices ? [false, -1] : false;
                    }
                } else {
                    let idx: number;
                    if (infoX.isComplex) {
                        if (infoY.isComplex) {
                            idx = DataHelper.firstIndexOfComplex(infoX.re, infoX.im, infoY.reArr, infoY.imArr);
                        } else {
                            idx = -1;
                        }
                    } else {
                        if (infoY.isComplex) {
                            // It is possible that some elements in y are real and
                            // matches x.
                            idx = DataHelper.firstIndexOfComplex(infoX.re, 0, infoY.reArr, infoY.imArr);
                        } else {
                            idx = DataHelper.firstIndexOf(infoX.re, infoY.reArr);
                        }
                    }
                    let flag = idx >= 0;
                    return outputIndices ? [flag, idx] : flag;
                }
            } else {
                // Calling unique to remove duplicates and obtain an ordered
                // list of elements.
                let [U, iy] = opUniqueInternal(infoY, false);
                let M = Tensor.zeros(infoX.originalShape, DType.LOGIC);
                let i: number;
                let n = infoX.reArr.length;
                let reX: ArrayLike<number> = infoX.reArr;
                let imX: ArrayLike<number>;
                let reU: ArrayLike<number> = U.realData;
                let imU: ArrayLike<number>;
                let reM = M.realData;
                if (outputIndices) {
                    // We need to store the indices.
                    let I = Tensor.zeros(infoX.originalShape, DTypeHelper.getDTypeOfIndices());
                    let reI = I.realData;
                    let idx: number;
                    if (infoX.isComplex) {
                        imX = infoX.imArr;
                        if (infoY.isComplex) {
                            // x is complex and y is complex
                            imU = U.imagData;
                            for (i = 0;i < n;i++) {
                                idx = DataHelper.binarySearchComplex(reX[i], imX[i], reU, imU);
                                if (idx >= 0) {
                                    reM[i] = 1;
                                    reI[i] = iy[idx];
                                } else {
                                    reM[i] = 0;
                                    reI[i] = -1;
                                }
                            }
                        } else {
                            // x is complex but y is real. We need to check if
                            // any real numbers in x appears in y.
                            for (i = 0;i < n;i++) {
                                if (imX[i] !== 0) {
                                    reI[i] = -1;
                                    reM[i] = 0;
                                } else {
                                    idx = DataHelper.binarySearch(reX[i], reU);
                                    if (idx >= 0) {
                                        reM[i] = 1;
                                        reI[i] = iy[idx];
                                    } else {
                                        reM[i] = 0;
                                        reI[i] = -1;
                                    }
                                }
                            }
                        }
                    } else {
                        if (infoY.isComplex) {
                            // x is real but y is complex
                            imU = U.imagData;
                            for (i = 0;i < n;i++) {
                                idx = DataHelper.binarySearchComplex(reX[i], 0, reU, imU);
                                if (idx >= 0) {
                                    reM[i] = 1;
                                    reI[i] = iy[idx];
                                } else {
                                    reM[i] = 0;
                                    reI[i] = -1;
                                }
                            }
                        } else {
                            // both x and y are real
                            for (i = 0;i < n;i++) {
                                idx = DataHelper.binarySearch(reX[i], reU);
                                if (idx >= 0) {
                                    reM[i] = 1;
                                    reI[i] = iy[idx];
                                } else {
                                    reM[i] = 0;
                                    reI[i] = -1;
                                }
                            }
                        }
                    }
                    return [M, I];
                } else {
                    // We do not need to store the indices.
                    if (infoX.isComplex) {
                        imX = infoX.imArr;
                        if (infoY.isComplex) {
                            imU = U.imagData;
                            for (i = 0;i < n;i++) {
                                reM[i] = DataHelper.binarySearchComplex(reX[i], imX[i], reU, imU) >= 0 ? 1 : 0;
                            }
                        } else {
                            // x is complex but y is real. We need to check if
                            // any real numbers in x appears in y.
                            for (i = 0;i < n;i++) {
                                if (imX[i] !== 0) {
                                    reM[i] = 0;
                                } else {
                                    reM[i] = DataHelper.binarySearch(reX[i], reU) >= 0 ? 1 : 0;
                                }
                            }
                        }
                    } else {
                        if (infoY.isComplex) {
                            imU = U.imagData;
                            for (i = 0;i < n;i++) {
                                reM[i] = DataHelper.binarySearchComplex(reX[i], 0, reU, imU) >= 0 ? 1 : 0;
                            }
                        } else {
                            for (i = 0;i < n;i++) {
                                reM[i] = DataHelper.binarySearch(reX[i], reU) >= 0 ? 1 : 0;
                            }
                        }
                    }
                    return M;
                }
            }
        }

        function opUnion(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
        function opUnion(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[], number[]];
        function opUnion(x: OpInput, y: OpInput, outputIndices: boolean = false): Tensor | [Tensor, number[], number[]] {
            let X = coreOp.flatten(x);
            let Y = coreOp.flatten(y);
            // To get a concise implementation we let unique() do the job.
            // It is possible have a longer and faster implementation without the concatenation.
            let Z = coreOp.concat([X, Y]);
            if (outputIndices) {
                let [U, iu] = opUniqueInternal(Tensor.analyzeOpInput(Z), false);
                let nX = X.size;
                // Restore index map.
                // If iu[i] >= nX, the corresponding element must belong to y.
                // Subtracting it by nX to obtain the correct index in y.
                let ix: number[] = [];
                let iy: number[] = [];
                for (let i = 0;i < iu.length;i++) {
                    if (iu[i] >= nX) {
                        iy.push(iu[i] - nX);
                    } else {
                        ix.push(iu[i]);
                    }
                }
                return [U, ix, iy];
            } else {
                return opUniqueInternal(Tensor.analyzeOpInput(Z), false)[0];
            }
        }

        function opIntersect(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
        function opIntersect(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[], number[]];
        function opIntersect(x: OpInput, y: OpInput, outputIndices: boolean = false): Tensor | [Tensor, number[], number[]] {
            let X = coreOp.flatten(x);
            let Y = coreOp.flatten(y);
            let Z: Tensor;
            let M: Tensor, I: Tensor, U: Tensor;
            let iRemaining: number[], iu: number[], ix: number[], iy: number[];
            let reI: ArrayLike<number>;
            let i: number;
            let outputDType = DTypeHelper.getWiderType(X.dtype, Y.dtype);
            if (X.size < Y.size) {
                if (outputIndices) {
                    [M, I] = opIsIn(X, Y, true);
                    iRemaining = coreOp.find(M); // indices of common elements in x
                    reI = I.realData; // indices of common elements in y
                    // Ensure the correct output data type.
                    Z = (<Tensor>X.get(iRemaining)).asType(outputDType, false);
                    // Remove duplicates.
                    [U, iu] = opUniqueInternal(Tensor.analyzeOpInput(Z), false);
                    // Restore index map.
                    iy = new Array(iu.length);
                    for (i = 0;i < iu.length;i++) {
                        iu[i] = iRemaining[iu[i]];
                        iy[i] = reI[iu[i]];
                    }
                    return [U, iu, iy];
                } else {
                    M = opIsIn(X, Y);
                    // Ensure the correct output data type.                    
                    Z = (<Tensor>X.get(M)).asType(outputDType, false);
                    return opUniqueInternal(Tensor.analyzeOpInput(Z), false)[0];
                }
            } else {
                // isIn() is more efficient when the first input is shorter
                // in length ((m + n) log n < (m + n) log m if m > n).
                if (outputIndices) {
                    [M, I] = opIsIn(Y, X, true);
                    iRemaining = coreOp.find(M); // indices of common elements in y
                    reI = I.realData; // indices of common elements in x
                    // Ensure the correct output data type.                    
                    Z = (<Tensor>Y.get(iRemaining)).asType(outputDType, false);
                    // Remove duplicates.
                    [U, iu] = opUniqueInternal(Tensor.analyzeOpInput(Z), false);
                    // Restore index map.
                    ix = new Array(iu.length);
                    for (i = 0;i < iu.length;i++) {
                        iu[i] = iRemaining[iu[i]];
                        ix[i] = reI[iu[i]];
                    }
                    return [U, ix, iu];
                } else {
                    M = opIsIn(Y, X);
                    // Ensure the correct output data type.
                    Z = (<Tensor>Y.get(M)).asType(outputDType, false);
                    return opUniqueInternal(Tensor.analyzeOpInput(Z), false)[0];
                }
            }
        }

        function opSetDiff(x: OpInput, y: OpInput, outputIndices?: false): Tensor;
        function opSetDiff(x: OpInput, y: OpInput, outputIndices: true): [Tensor, number[]];
        function opSetDiff(x: OpInput, y: OpInput, outputIndices: boolean = false): Tensor | [Tensor, number[]] {
            let X = coreOp.flatten(x);
            let Y = coreOp.flatten(y);
            let M: Tensor;
            // Masks all the elements that appear in y.
            M = opIsIn(X, Y);
            logicCompOp.not(M, true);
            // Obtain the indices for the remaining elements.
            let iRemaining = coreOp.find(M);
            // Call unique to remove duplicates.
            let [U, iy] = opUniqueInternal(Tensor.analyzeOpInput(X.get(iRemaining)), false);
            if (outputIndices) {
                for (let i = 0;i < iy.length;i++) {
                    iy[i] = iRemaining[iy[i]];
                }
                return [U, iy];
            } else {
                return U;
            }
        }

        return {
            unique: opUnique,
            isin: opIsIn,
            union: opUnion,
            intersect: opIntersect,
            setdiff: opSetDiff
        };
    }

}
