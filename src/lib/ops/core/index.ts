import { ICoreOpProvider } from '../definition';
import { Tensor } from '../../tensor';
import { ShapeHelper } from '../../helper/shapeHelper';
import { DataHelper } from '../../helper/dataHelper';
import { DType, OutputDTypeResolver, DTypeHelper } from '../../dtype';
import { DataBlock } from '../../storage';
import { OpInput } from '../../commonTypes';

export class CoreOpProviderFactory {
    public static create(): ICoreOpProvider {

        const opReshape = (x: OpInput, shape: number[]) => {
            if (x instanceof Tensor) {
                return x.getReshapedCopy(shape);
            } else {
                let X = Tensor.toTensor(x);
                return X.reshape(shape);
            }
        };

        const opSqueeze = (x: OpInput) => {
            if (x instanceof Tensor) {
                return x.getReshapedCopy(ShapeHelper.getSqueezedShape(x.shape));
            } else {
                let X = Tensor.toTensor(x);
                return X.reshape(ShapeHelper.getSqueezedShape(X.shape));
            }
        };

        const opPrependAxis = (x: OpInput) => {
            let t = x instanceof Tensor ? x : Tensor.toTensor(x);
            let shape = t.shape;
            shape.unshift(1);
            return x instanceof Tensor ? t.getReshapedCopy(shape) : t.reshape(shape);            
        };

        const opAppendAxis = (x: OpInput) => {
            let t = x instanceof Tensor ? x : Tensor.toTensor(x);
            let shape = t.shape;
            shape.push(1);
            return x instanceof Tensor ? t.getReshapedCopy(shape) : t.reshape(shape);
        };

        const opFlatten = (x: OpInput) => {
            return x instanceof Tensor 
                ? x.getReshapedCopy([-1])
                : Tensor.toTensor(x).reshape([-1]);
        };

        const opVec = (x: OpInput) => {
            return x instanceof Tensor 
                ? x.getReshapedCopy([-1, 1])
                : Tensor.toTensor(x).reshape([-1, 1]);
        };

        const opTile = (x: OpInput, repeats: number[]): Tensor => {
            let X = x instanceof Tensor ? x : Tensor.toTensor(x);
            let shapeX = X.shape;
            // Prepend repeats with ones if its length is less than the number
            // dimensions of x.
            if (repeats.length < shapeX.length) {
                // pad repeats
                let repeatsOld = repeats;
                repeats = new Array(shapeX.length);
                let i = 0, diff = shapeX.length - repeatsOld.length;
                for (;i < diff;i++) {
                    repeats[i] = 0;
                }
                for (;i < shapeX.length;i++) {
                    repeats[i] = repeatsOld[i - diff];
                }
            }

            // Determines the shape of the output tensor B and create it.
            let nExtraDims = repeats.length - shapeX.length;            
            let shapeY = new Array(repeats.length);
            let i = 0;
            for (;i < nExtraDims;i++) {
                shapeY[i] = repeats[i];
            }
            for (;i < repeats.length;i++) {
                shapeY[i] = repeats[i] * shapeX[i - nExtraDims];
            }
            let Y = Tensor.zeros(shapeY, X.dtype);
            
            // Tile
            tileImpl(X.realData, shapeX, repeats, Y.realData, shapeY);
            if (X.hasComplexStorage()) {
                Y.ensureComplexStorage();
                tileImpl(X.imagData, shapeX, repeats, Y.imagData, shapeY);                
            }

            return Y;
        };

        const tileImpl = (X: DataBlock, shapeX: number[], repeats: number[], Y: DataBlock, shapeY: number[]): void => {
            // Let shapeX = [n_1, n_2, ... n_D], repeats = [r_1, r_2, ... r_K],
            // where K >= D. Then shapeB is
            //  [r_1, r_2, ..., r_{K-D+1}*n_1, r_{k-D+2}*n_2, ..., r_K*n_D]

            // Special case: X is a scalar
            if (X.length === 1) {
                // We just need to full Y with the same values
                for (let i = 0;i < Y.length;i++) {
                    Y[i] = X[0];
                }
                return;
            }
         
            // General case: we first tile X with repeats
            // [1, r_{K-D+2}, r_{K-D+3}, ..., r_K] to obtain Y0, and then repeat
            // Y0 r1*r2*...*r_{K-D+1} times.

            // 1. Tile X into Y0
            let nExtraDims = shapeY.length - shapeX.length;
            let stridesY = ShapeHelper.computeStrides(shapeY);
            let nCopy = repeats[nExtraDims];
            for (let i = 0;i < nExtraDims;i++) {
                nCopy *= repeats[i];
            }
            // Check if repeats = [r_1, r_2, ..., r_{K-D+1}, 1, ..., 1]
            let directCopy = true;
            for (let i = nExtraDims + 1;i < repeats.length;i++) {
                if (repeats[i] !== 1) {
                    directCopy = false;
                    break;
                }
            }
            if (directCopy) {
                // Here repeats = [r_1, r_2, ..., r_{K-D+1}, 1, ..., 1] and
                // Y0 can be obtained by directly copying X.
                for (let i = 0;i < X.length;i++) {
                    Y[i] = X[i];
                }
            } else {
                // TODO: if shapeX has lots of trailing ones, the following
                // implementation is not very efficient.
                let stridesX = ShapeHelper.computeStrides(shapeX);
                let maxLevel = shapeX.length - 1;
                let doTile = (level: number, offsetX: number, offsetY: number): void => {
                    if (level === maxLevel) {
                        // last level
                        for (let j = 0;j < repeats[nExtraDims + level];j++) {
                            for (let i = 0;i < shapeX[level];i++) {
                                Y[offsetY + i] = X[offsetX + i];
                            }
                            offsetY += stridesY[nExtraDims + level] * shapeX[level];
                        }
                    } else {
                        // recursive calling
                        let offsetX0 = offsetX;
                        for (let j = 0;j < repeats[nExtraDims + level];j++) {
                            offsetX = offsetX0;
                            for (let i = 0;i < shapeX[level];i++) {
                                doTile(level + 1, offsetX, offsetY);
                                offsetX += stridesX[level];
                                offsetY += stridesY[nExtraDims + level];
                            }
                        }
                    }
                };
                let tmp = repeats[nExtraDims];
                repeats[nExtraDims] = 1;
                doTile(0, 0, 0);
                repeats[nExtraDims] = tmp;
            }

            // 2. Repeat Y0 r1*r2*...*r_{K-D+1} times
            let offsetY = stridesY[nExtraDims] * shapeX[0],
                strideCopy = offsetY;
            for (let i = 1;i < nCopy;i++) {
                for (let j = 0;j < strideCopy;j++) {
                    Y[offsetY + j] = Y[j];
                }
                offsetY += strideCopy;
            }
        };

        const opConcat = (inputs: OpInput[], axis: number = 0): Tensor => {
            if (inputs.length === 0) {
                throw new Error('At least one input expected.');
            }
            let t = inputs[0];
            if (inputs.length === 1) {
                return t instanceof Tensor ? t.copy() : Tensor.toTensor(t);
            }
            // Unify inputs, check shapes and determine the final shape.
            let tensors = new Array<Tensor>(inputs.length);
            let shapes = new Array<number[]>(inputs.length);
            let strides = new Array<number[]>(inputs.length);
            let first = t instanceof Tensor ? t : Tensor.toTensor(t);
            tensors[0] = first;
            shapes[0] = first.shape;
            strides[0] = first.strides;
            let finalShape = first.shape;
            let finalDType = first.dtype;
            let needComplexStorage = first.hasComplexStorage();
            let ndim = finalShape.length;
            for (let i = 1;i < inputs.length;i++) {
                let curInput = inputs[i];
                let curTensor = curInput instanceof Tensor ? curInput : Tensor.toTensor(curInput);
                let curShape = curTensor.shape;
                if (curShape.length !== ndim) {
                    throw new Error('Tensors being concatenated must share the same number of dimensions.')
                }
                for (let j = 0;j < ndim;j++) {
                    if (j === axis) {
                        finalShape[axis] += curShape[axis];
                    } else {
                        if (curShape[j] !== finalShape[j]) {
                            throw new Error('Tensors being concatenated must have matching dimensions ' +
                                'except the dimension being concatenated.')
                        }
                    }
                }
                finalDType = DTypeHelper.getWiderType(finalDType, curTensor.dtype);
                needComplexStorage = needComplexStorage || curTensor.hasComplexStorage();
                tensors[i] = curTensor;
                shapes[i] = curShape;
                strides[i] = curTensor.strides;
            }
            let Y = Tensor.zeros(finalShape, finalDType);
            if (needComplexStorage) {
                Y.ensureComplexStorage();
            }
            // concat
            if (axis === 0) {
                // Concatenating along the first dimension is straightforward.
                let offset = 0;
                for (let i = 0;i < tensors.length;i++) {
                    DataHelper.copy(tensors[i].realData, Y.realData, offset);
                    if (tensors[i].hasComplexStorage()) {
                        DataHelper.copy(tensors[i].imagData, Y.imagData, offset);
                    }
                    offset += tensors[i].size;
                }
            } else {
                // Concatenating along other dimensions requires some extra
                // work.
                let finalStrides = Y.strides;
                let axisOffset = 0;
                for (let i = 0;i < tensors.length;i++) {
                    copyWithAxisOffset(tensors[i].realData, shapes[i], strides[i],
                        Y.realData, finalStrides, axis, axisOffset);
                    if (tensors[i].hasComplexStorage()) {
                        copyWithAxisOffset(tensors[i].imagData, shapes[i], strides[i],
                            Y.imagData, finalStrides, axis, axisOffset);
                    }
                    axisOffset += shapes[i][axis];
                }
            }
            return Y;
        };

        function copyWithAxisOffset(source: ArrayLike<number>, sourceShape: number[],
                                    sourceStrides: number[], target: DataBlock,
                                    targetStrides: number[], axis: number,
                                    axisOffset: number): void {
            const maxLevel = sourceShape.length - 1;
            const doCopy = (level: number, offsetS: number, offsetT: number): void => {
                if (level === maxLevel) {
                    for (let i = 0;i < sourceShape[level];i++) {
                        target[offsetT + i] = source[offsetS + i];
                    }
                } else {
                    for (let i = 0;i < sourceShape[level];i++) {
                        doCopy(level + 1, offsetS, offsetT);
                        offsetS += sourceStrides[level];
                        offsetT += targetStrides[level];
                    }
                }
            };
            doCopy(0, 0, targetStrides[axis] * axisOffset);
        }

        const opReal = (x: OpInput) => {
            return x instanceof Tensor ? x.real() : Tensor.toTensor(x).real();
        };

        const opImag = (x: OpInput) => {
            return x instanceof Tensor ? x.imag() : Tensor.toTensor(x).imag();
        };

        const opIsReal = (x: OpInput) => {
            let t = x instanceof Tensor ? x : Tensor.toTensor(x);
            if (!t.hasComplexStorage()) return false;
            return DataHelper.isArrayAllZeros(t.imagData);
        };

        const opIsNaN = (x: OpInput) => {
            let t = x instanceof Tensor ? x : Tensor.toTensor(x);
            let result = Tensor.zeros(t.shape, DType.LOGIC);
            let reX = t.realData;
            let imX = t.imagData;
            let reY = result.realData;
            if (t.hasComplexStorage()) {
                for (let i = 0;i < t.size;i++) {
                    reY[i] = (isNaN(reX[i]) || isNaN(imX[i])) ? 1 : 0;
                }
            } else {
                for (let i = 0;i < t.size;i++) {
                    reY[i] = isNaN(reX[i]) ? 1 : 0;
                }
            }
            return result;
        };

        const opIsInf = (x: OpInput) => {
            let t = x instanceof Tensor ? x : Tensor.toTensor(x);
            let result = Tensor.zeros(t.shape, DType.LOGIC);
            let reX = t.realData;
            let imX = t.imagData;
            let reY = result.realData;
            if (t.hasComplexStorage()) {
                for (let i = 0;i < t.size;i++) {
                    reY[i] = !(isFinite(reX[i]) && isFinite(imX[i])) && !(isNaN(reX[i]) || isNaN(imX[i])) ? 1 : 0;
                }
            } else {
                for (let i = 0;i < t.size;i++) {
                    reY[i] = !isFinite(reX[i]) && !isNaN(reX[i]) ? 1 : 0;
                }
            }
            return result;
        };

        const opFind = (x: OpInput, f?: (re: number, im: number) => boolean) => {
            let t = x instanceof Tensor ? x : Tensor.toTensor(x);
            let indices: number[];
            let reX = t.realData;
            let imX = t.imagData;
            if (f) {
                if (t.hasComplexStorage()) {
                    indices = DataHelper.findWithCallbackComplex(reX, imX, f);
                } else {
                    indices = DataHelper.findWithCallbackReal(reX, f);
                }
            } else {
                if (t.hasComplexStorage()) {
                    indices = DataHelper.findComplex(reX, imX);
                } else {
                    indices = DataHelper.findReal(reX);
                }
            }
            return indices;
        };

        const opLinspace = (x1: number, x2: number, n: number) => {
            if (n < 1) {
                throw new Error('Number of samples n must be greater or equal to 1.');
            }
            if (!isFinite(x1) || !isFinite(x2) || !isFinite(n)) {
                throw new Error('Inputs must be finite.');
            }
            n = Math.floor(n);
            let y: Tensor;
            if (n === 1) {
                y = Tensor.scalar(x2);
            } else {
                y = Tensor.zeros([n]);
                let n1 = n - 1;
                let delta = x2 / n1 - x1 / n1;
                y.realData[0] = x1;
                y.realData[n1] = x2; 
                for (let i = 1; i < n - 1; i++) {
                    y.realData[i] = x1 + i * delta;
                }
            }
            return y;
        };

        const opLogspace = (x1: number, x2: number, n: number, base: number = 10): Tensor => {
            if (base <= 0 || !isFinite(base)) {
                throw new Error('Base must be a finite nonnegative real number.');
            }
            let Y = opLinspace(x1, x2, n);
            let re = Y.realData;
            for (let i = 0;i < n;i++) {
                re[i] = Math.pow(base, re[i]);
            }
            return Y;
        };

        return {
            reshape: opReshape,
            flatten: opFlatten,
            vec: opVec,
            squeeze: opSqueeze,
            tile: opTile,
            concat: opConcat,
            prependAxis: opPrependAxis,
            appendAxis: opAppendAxis,
            linspace: opLinspace,
            logspace: opLogspace,
            real: opReal,
            imag: opImag,
            isreal: opIsReal,
            isnan: opIsNaN,
            isinf: opIsInf,
            find: opFind
        };

    }
}