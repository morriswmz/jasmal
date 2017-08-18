import { IPolynomialOpProvider } from './definition';
import { Tensor } from "../../tensor";
import { OutputDTypeResolver, DTypeHelper } from '../../dtype';
import { PolynomialFunction } from './polyfun';
import { ComplexNumber } from '../../complexNumber';
import { OpInput, OpOutput, OpInputType, DataBlock } from '../../commonTypes';
import { IMatrixOpProvider } from '../matrix/definition';

export class PolynomialOpProviderFactory {

    public static create(matOp: IMatrixOpProvider): IPolynomialOpProvider {

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
                        [re, im] = PolynomialFunction.evalPolyCC(reP, imP, infoX.re, infoX.im);
                    } else {
                        [re, im] = PolynomialFunction.evalPolyCR(reP, imP, infoX.re);
                    }
                } else {
                    if (infoX.isComplex) {
                        [re, im] = PolynomialFunction.evalPolyRC(reP, infoX.re, infoX.im);
                    } else {
                        re = PolynomialFunction.evalPolyRR(reP, infoX.re);
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
                            [reY[i], imY[i]] = PolynomialFunction.evalPolyCC(reP, imP, reX[i], imX[i]);
                        }
                    } else {
                        for (i = 0;i < n;i++) {
                            [reY[i], imY[i]] = PolynomialFunction.evalPolyCR(reP, imP, reX[i]);
                        }
                    }
                } else {
                    if (infoX.isComplex) {
                        imX = infoX.imArr;
                        Y.ensureComplexStorage();
                        imY = Y.imagData;
                        for (i = 0;i < n;i++) {
                            [reY[i], imY[i]] = PolynomialFunction.evalPolyRC(reP, reX[i], imX[i]);
                        }
                    } else {
                        for (i = 0;i < n;i++) {
                            reY[i] = PolynomialFunction.evalPolyRR(reP, reX[i]);
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
                Y = <Tensor>matOp.matmul(Y, x);
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
        }

        return {
            polyval: opPolyval,
            polyvalm: opPolyvalm
        };

    }

}