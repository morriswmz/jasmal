import { IDataOpProvider } from '../definition';
import { Tensor } from '../../tensor';
import { NOT_IMPLEMENTED } from '../../constant';
import { OpInput, OpOutput, OpOutputWithIndex } from '../../commonTypes';
import { DataFunction } from './datafun';
import { ComplexNumber } from '../../complexNumber';
import { ShapeHelper } from '../../helper/shapeHelper';
import { DataBlock } from '../../storage';
import { DType, OutputDTypeResolver } from '../../dtype';

type RealDataFunction = (reX: ArrayLike<number>, offset: number, stride: number, n: number) => number;
type ComplexDataFunction1 = (reX: ArrayLike<number>, imX: ArrayLike<number>, offset: number, stride: number, n: number)
                              => number;
type ComplexDataFunction2 = (reX: ArrayLike<number>, imX: ArrayLike<number>, offset: number, stride: number, n: number)
                              => [number, number];

export class DataOpProviderFactory {

    public static create(): IDataOpProvider {

        function reduceAlongDimension(X: Tensor, axis: number, keepDims: boolean,
                                      outputDType: DType,
                                      isOutputComplex: false,
                                      fReal: RealDataFunction,
                                      fComplex?: ComplexDataFunction1): Tensor;
        function reduceAlongDimension(X: Tensor, axis: number, keepDims: boolean,
                                      outputDType: DType,
                                      isOutputComplex: true,
                                      fReal: RealDataFunction,
                                      fComplex?: ComplexDataFunction2): Tensor;
        function reduceAlongDimension(X: Tensor, axis: number, keepDims: boolean,
                                      outputDType: DType,
                                      isOutputComplex: boolean,
                                      fReal: RealDataFunction,
                                      fComplex?: ComplexDataFunction1 | ComplexDataFunction2): Tensor {
            if (axis >= X.ndim) {
                throw new Error(`Invalid axis number ${axis}.`);
            }
            let shapeX = X.shape;
            let shapeY = shapeX.slice();
            shapeY[axis] = 1;
            let Y = Tensor.zeros(shapeY, outputDType);
            let stridesX = X.strides;
            let stridesY = Y.strides;
            let maxLevel = X.ndim - 1;
            let stride = stridesX[axis];
            let n = shapeX[axis];
            let reX: ArrayLike<number>, imX: ArrayLike<number>;
            let reY: DataBlock, imY: DataBlock;
            if (X.hasComplexStorage()) {
                if (fComplex == undefined) {
                    throw new Error('Operation not permitted for complex matrices.');
                }
                if (isOutputComplex) {
                    Y.ensureComplexStorage();
                    reX = X.realData;
                    reY = Y.realData;
                    reY = Y.realData;
                    imY = Y.imagData;
                    const doCalc = (level: number, offsetX: number, offsetY): void => {
                        if (level === maxLevel) {
                            for (let j = 0;j < shapeY[level];j++) {
                                [reY[offsetY], imY[offsetY]] = (<ComplexDataFunction2>fComplex)(reX, imX, offsetX, stride, n);
                                offsetX++;
                                offsetY++;
                            }
                        } else {
                            for (let i = 0;i < shapeY[level];i++) {
                                doCalc(level + 1, offsetX, offsetY);
                                offsetX += stridesX[level];
                                offsetY += stridesY[level];
                            }
                        }
                    };
                    doCalc(0, 0, 0);
                } else {
                    reX = X.realData;
                    reY = Y.realData;
                    reY = Y.realData;
                    const doCalc = (level: number, offsetX: number, offsetY): void => {
                        if (level === maxLevel) {
                            for (let j = 0;j < shapeY[level];j++) {
                                reY[offsetY] = (<ComplexDataFunction1>fComplex)(reX, imX, offsetX, stride, n);
                                offsetX++;
                                offsetY++;
                            }
                        } else {
                            for (let i = 0;i < shapeY[level];i++) {
                                doCalc(level + 1, offsetX, offsetY);
                                offsetX += stridesX[level];
                                offsetY += stridesY[level];
                            }
                        }
                    };
                    doCalc(0, 0, 0);
                }
            } else {
                reX = X.realData;
                reY = Y.realData;
                const doCalc = (level: number, offsetX: number, offsetY): void => {
                    if (level === maxLevel) {
                        for (let j = 0;j < shapeY[level];j++) {
                            reY[offsetY] = fReal(reX, offsetX, stride, n);
                            offsetX++;
                            offsetY++;
                        }
                    } else {
                        for (let i = 0;i < shapeY[level];i++) {
                            doCalc(level + 1, offsetX, offsetY);
                            offsetX += stridesX[level];
                            offsetY += stridesY[level];
                        }
                    }
                };
                doCalc(0, 0, 0);
            }
            if (!keepDims) {
                Y.reshape(ShapeHelper.getSqueezedShape(shapeY));
            }
            return Y;
        };

        const opMin = (x: OpInput, axis: number = -1): OpOutputWithIndex => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.hasComplexStorage()) {
                throw new Error('Minimum is not defined for complex numbers.');
            }
            if (axis < 0 || (axis === 0 && X.ndim === 1)) {
                return DataFunction.min(X.realData);
            } else {
                throw new Error('Not implemented.');
            }
        };

        const opMax = (x: OpInput, axis: number = -1): OpOutputWithIndex => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.hasComplexStorage()) {
                throw new Error('Maximum is not defined for complex numbers.');
            }
            if (axis < 0 || (axis === 0 && X.ndim === 1)) {
                return DataFunction.max(X.realData);
            } else {
                throw new Error('Not implemented.');
            }
        };


        const opSum = (x: OpInput, axis: number = -1, keepDims: boolean = false): OpOutput => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (axis < 0 || (axis === 0 && X.ndim === 1)) {
                if (X.hasComplexStorage()) {
                    return new ComplexNumber(DataFunction.sum(X.realData),
                                             DataFunction.sum(X.imagData));
                } else {
                    return DataFunction.sum(X.realData);
                }
            } else {
                return reduceAlongDimension(X, axis, keepDims, 
                    OutputDTypeResolver.uOnlyLogicToFloat(X.dtype, X.hasComplexStorage()),
                    true, DataFunction.sum, (reX, imX, offset, stride, n) => {
                        return [DataFunction.sum(reX, offset, stride, n),
                                DataFunction.sum(imX, offset, stride, n)]
                    });
            }
        };

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

        const opProd = (x: OpInput, axis: number = -1, keepDims: boolean = false): OpOutput => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (axis < 0 || (axis === 0 && X.ndim === 1)) {
                if (X.hasComplexStorage()) {
                    let [re, im] = DataFunction.cprod(X.realData, X.imagData);
                    return new ComplexNumber(re, im);
                } else {
                    return DataFunction.prod(X.realData);
                }
            } else {
                return reduceAlongDimension(X, axis, keepDims, 
                    OutputDTypeResolver.uOnlyLogicToFloat(X.dtype, X.hasComplexStorage()),
                    true, DataFunction.prod, DataFunction.cprod);
            }
        };

        const opMean = (x: OpInput, axis: number = -1, keepDims: boolean = false): OpOutput => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (axis < 0 || (axis === 0 && X.ndim === 1)) {
                if (X.hasComplexStorage()) {
                    return new ComplexNumber(DataFunction.sum(X.realData) / X.size,
                                             DataFunction.sum(X.imagData) / X.size);
                } else {
                    return DataFunction.sum(X.realData) / X.size;
                }
            } else {
                return reduceAlongDimension(X, axis, keepDims,
                    OutputDTypeResolver.uOnlyLogicToFloat(X.dtype, X.hasComplexStorage()),
                    true, (reX, offset, stride, n) => {
                        return DataFunction.sum(reX, offset, stride, n) / n;
                    }, (reX, imX, offset, stride, n) => {
                        return [DataFunction.sum(reX, offset, stride, n) / n,
                                DataFunction.sum(imX, offset, stride, n) / n];
                    });
            }
        };

        const opMedian = (x: OpInput, axis: number = -1, keepDims: boolean = false): OpOutput => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (X.hasComplexStorage()) {
                throw new Error('Cannot compute median for complex tensors.');
            }
            if (axis < 0 || (axis === 0 && X.ndim === 1)) {
                return DataFunction.median(X.realData);
            } else {
                return reduceAlongDimension(X, axis, keepDims,
                    OutputDTypeResolver.uOnlyLogicToFloat(X.dtype, X.hasComplexStorage()),
                    true, DataFunction.median);
            }
        };

        const opVar = (x: OpInput, axis: number = -1, keepDims: boolean = false): OpOutput => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (axis < 0 || (axis === 0 && X.ndim === 1)) {
                if (X.hasComplexStorage()) {
                    return DataFunction.cvar(X.realData, X.imagData);
                } else {
                    return DataFunction.var(X.realData);
                }
            } else {
                return reduceAlongDimension(X, axis, keepDims, 
                    OutputDTypeResolver.uOnlyLogicToFloat(X.dtype, X.hasComplexStorage()),
                    false, DataFunction.var, DataFunction.cvar);
            }
        };

        const opStd = (x: OpInput, axis: number = -1, keepDims: boolean = false): OpOutput => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (axis < 0 || (axis === 0 && X.ndim === 1)) {
                if (X.hasComplexStorage()) {
                    return Math.sqrt(DataFunction.cvar(X.realData, X.imagData));
                } else {
                    return Math.sqrt(DataFunction.var(X.realData));
                }
            } else {
                let f: ComplexDataFunction1 = DataFunction.cvar;
                return reduceAlongDimension(X, axis, keepDims, 
                    OutputDTypeResolver.uOnlyLogicToFloat(X.dtype, X.hasComplexStorage()),
                    false, (reX, offset, stride, n) => {
                        return Math.sqrt(DataFunction.var(reX, offset, stride, n));
                    }, (reX, imX, offset, stride, n) => {
                        return Math.sqrt(DataFunction.cvar(reX, imX, offset, stride, n)); });
            }
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
            var: opVar
        };

    }

}