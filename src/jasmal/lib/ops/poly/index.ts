import { IPolynomialOpProvider } from './definition';
import { Tensor } from "../../core/tensor";
import { OutputDTypeResolver, DTypeHelper } from '../../core/dtype';
import { PolynomialEvaluator } from './polyfun';
import { ComplexNumber } from '../../core/complexNumber';
import { OpInput, OpOutput, OpInputType, DataBlock } from '../../commonTypes';
import { IMatrixOpProvider } from '../matrix/definition';
import { ICoreOpProvider } from '../core/definition';
import { CMath } from '../../math/cmath';
import { DataHelper } from "../../helper/dataHelper";
import { IJasmalModuleFactory, JasmalOptions } from '../../jasmal';

export class PolynomialOpProviderFactory implements IJasmalModuleFactory<IPolynomialOpProvider> {
    
    constructor(private _coreOp: ICoreOpProvider, private _matOp: IMatrixOpProvider) {
    }
    
    public create(_options: JasmalOptions): IPolynomialOpProvider {

        const coreOp = this._coreOp;
        const matOp = this._matOp;
        
        const opPolyval = (p: OpInput, x: OpInput, inPlace: boolean = false): OpOutput => {
            let infoP = Tensor.analyzeOpInput(p);
            if (infoP.originalShape.length !== 1) {
                throw new Error('Vector expected for p.');
            }
            let infoX = Tensor.analyzeOpInput(x);
            if (infoP.isInputScalar) {
                infoP.reArr = [infoP.re];
                infoP.imArr = [infoP.im];
            }
            let reP = infoP.reArr, imP = infoP.imArr;
            if (infoX.isInputScalar) {
                let re: number, im: number = 0;
                if (infoP.isComplex) {
                    if (infoX.isComplex) {
                        [re, im] = PolynomialEvaluator.evalPolyCC(reP, imP, infoX.re, infoX.im);
                    } else {
                        [re, im] = PolynomialEvaluator.evalPolyCR(reP, imP, infoX.re);
                    }
                } else {
                    if (infoX.isComplex) {
                        [re, im] = PolynomialEvaluator.evalPolyRC(reP, infoX.re, infoX.im);
                    } else {
                        re = PolynomialEvaluator.evalPolyRR(reP, infoX.re);
                    }
                }
                return im === 0 ? re : new ComplexNumber(re, im);
            } else {
                let outputDType = OutputDTypeResolver.bWiderWithLogicToInt(
                    infoP.originalDType, infoP.isComplex, infoX.originalDType, infoX.isComplex);
                let Y: Tensor;
                if (inPlace) {
                    if (infoX.originalType !== OpInputType.Tensor) {
                        throw new Error('Cannot perform in-place operations when the operand is not a tensor.')
                    } else if (DTypeHelper.isWiderType(outputDType, infoX.originalDType)) {
                        throw new Error('Cannot perform in-place operations because the output data type is incompatible.');
                    }
                    Y = <Tensor>x;
                    Y.ensureUnsharedLocalStorage();
                } else {
                    Y = Tensor.zeros(infoX.originalShape, outputDType);
                }
                let reX: DataBlock = infoX.reArr;
                let imX: DataBlock;
                let reY: DataBlock = Y.realData;
                let imY: DataBlock;
                let i: number, n = Y.size;
                if (infoP.isComplex) {
                    Y.ensureComplexStorage();
                    imY = Y.imagData;
                    if (infoX.isComplex) {
                        imX = infoX.imArr;
                        for (i = 0;i < n;i++) {
                            [reY[i], imY[i]] = PolynomialEvaluator.evalPolyCC(reP, imP, reX[i], imX[i]);
                        }
                    } else {
                        for (i = 0;i < n;i++) {
                            [reY[i], imY[i]] = PolynomialEvaluator.evalPolyCR(reP, imP, reX[i]);
                        }
                    }
                } else {
                    if (infoX.isComplex) {
                        imX = infoX.imArr;
                        Y.ensureComplexStorage();
                        imY = Y.imagData;
                        for (i = 0;i < n;i++) {
                            [reY[i], imY[i]] = PolynomialEvaluator.evalPolyRC(reP, reX[i], imX[i]);
                        }
                    } else {
                        for (i = 0;i < n;i++) {
                            reY[i] = PolynomialEvaluator.evalPolyRR(reP, reX[i]);
                        }
                    }
                }
                return Y;
            }
        };

        const opPolyvalm = (p: OpInput, x: OpInput): OpOutput => {
            if (typeof x === 'number' || x instanceof ComplexNumber) {
                return opPolyval(p, x);
            }
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.ndim !== 2) {
                throw new Error('Matrix expected.');
            }
            let [m, n] = X.shape;
            if (m !== n) {
                throw new Error('Square matrix expected.');
            }
            let isXComplex = X.hasNonZeroComplexStorage();
            let infoP = Tensor.analyzeOpInput(p);
            if (infoP.originalShape.length !== 1) {
                throw new Error('Vector expected.');
            }
            if (infoP.isInputScalar) {
                infoP.reArr = [infoP.re];
                infoP.imArr = [infoP.im];
            }
            let outputDType = OutputDTypeResolver.bWiderWithLogicToInt(
                infoP.originalDType, infoP.isComplex, X.dtype, isXComplex);
            let Y: Tensor = Tensor.zeros([m, m], outputDType);
            let i: number, k: number;
            let reY = Y.realData;
            let imY: DataBlock = [];
            for (k = 0;k < m;k++) {
                reY[k * m + k] = infoP.reArr[0];
            }
            if (infoP.isComplex) {
                Y.ensureComplexStorage();
                imY = Y.imagData;
                for (k = 0;k < m;k++) {
                    imY[k * m + k] = infoP.imArr[0];
                }
            }
            for (i = 1;i < infoP.reArr.length;i++) {
                // multiply
                Y = matOp.matmul(Y, x);
                // add
                reY = Y.realData;
                for (k = 0;k < m;k++) {
                    reY[k * m + k] += infoP.reArr[i];
                }
                if (infoP.isComplex) {
                    Y.ensureComplexStorage();
                    imY = Y.imagData;
                    for (k = 0;k < m;k++) {
                        imY[k * m + k] += infoP.imArr[i];
                    }
                }
            }
            return Y;
        };

        const opPolyfit = (x: OpInput, y: OpInput, n: number): Tensor => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            let Y = y instanceof Tensor ? y : Tensor.toTensor(y);
            if (X.size !== Y.size) {
                throw new Error('x and y must have the same size.');
            }
            return matOp.linsolve(matOp.vander(X, n + 1), Y).reshape([-1]);
        };

        const opRoots = (p: OpInput): Tensor => {
            let P = p instanceof Tensor ? p : Tensor.toTensor(p);
            if (P.ndim > 1) {
                throw new Error('Vector expected.');
            }
            if (!DataHelper.isArrayAllFinite(P.realData) || (P.hasComplexStorage() && !DataHelper.isArrayAllFinite(P.imagData))) {
                throw new Error('Coefficients cannot contain NaN or Infinity.')
            }
            // remove zeros in the beginning
            let idxNonZero = coreOp.find(P);
            if (idxNonZero.length === 0) {
                throw new Error('Coefficients cannot be all zeros.');
            }
            // record the number of trailing zeros
            let nDeg = P.size - 1;            
            let nTrailingZeros = nDeg - idxNonZero[idxNonZero.length - 1];
            if (idxNonZero.length === 1) {
                if (nTrailingZeros === 0) {
                    throw new Error('The effective degree must be at least one.');
                } else {
                    // all zeros
                    return Tensor.zeros([nTrailingZeros]);
                }
            }
            nDeg = nDeg - idxNonZero[0] - nTrailingZeros; // effective degree
            // remove zeros in the beginning and the end
            // TODO: handle small coefficients to avoid inf
            P = <Tensor>P.get(`${idxNonZero[0]}:${idxNonZero[idxNonZero.length - 1] + 1}`);
            let res: Tensor;
            if (P.size === 2) {
                // linear
                if (P.hasNonZeroComplexStorage()) {
                    let [re, im] = CMath.cdivCC(-P.realData[1], -P.imagData[1], P.realData[0], P.imagData[0]);
                    res = Tensor.scalar(re, im);
                } else {
                    res = Tensor.scalar(-P.realData[1] / P.realData[0]);
                }
            } else {
                // use the eigenvalue method
                // construct the companion matrix
                let A = Tensor.zeros([nDeg, nDeg]);
                let reA = A.realData;
                let reP = P.realData;
                let i;
                if (P.hasNonZeroComplexStorage()) {
                    A.ensureComplexStorage();
                    let imA = A.imagData;
                    let imP = P.imagData;
                    for (i = 0;i < nDeg;i++) {
                        [reA[i], imA[i]] = CMath.cdivCC(-reP[i + 1], -imP[i + 1], reP[0], imP[0]);
                    }  
                } else {
                    for (i = 0;i < nDeg;i++) {
                        reA[i] = -reP[i + 1] / reP[0];
                    }     
                }
                for (i = 1;i < nDeg;i++) {
                    reA[i * nDeg + i - 1] = 1;
                }
                res = matOp.eig(A, true);
            }
            return nTrailingZeros > 0 ? coreOp.concat([Tensor.zeros([nTrailingZeros]), res]) : res;
        };

        return {
            polyval: opPolyval,
            polyvalm: opPolyvalm,
            polyfit: opPolyfit,
            roots: opRoots
        };

    }

}
