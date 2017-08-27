import { DType } from '../../../dtype';

/**
 * Main template for reduction operations.
 * Supported configurations:
 *  NO_COMPLEX_INPUT
 *  OUTPUT_INDICES
 *  OUTPUT_R_COMPLEX
 *  OUTPUT_C_COMPLEX
 */
export const REDUCTION_OP_TEMPLATE =
`'use strict';
$Dependencies
return function(x, axis, keepDims) {
    if (axis == undefined) axis = -1;
    var keepDims = keepDims || false;
    var X = x instanceof Tensor ? x : Tensor.toTensor(x);
    if (axis !== -1 && ((axis | 0) !== axis || axis >= X.ndim)) {
        throw new Error('Invalid axis number ' + axis + '.');
    }
    var isInputComplex = X.hasNonZeroComplexStorage();
#if NO_COMPLEX_INPUT
    if (isInputComplex) {
        throw new Error('Complex input is not supported.');
    }
#endif
    var tmp;
    var outputDType = outputDTypeResolver(X.dtype, isInputComplex);
    var indexDType = ObjectHelper.hasTypedArraySupport() ? ${DType.INT32} : ${DType.FLOAT64};
    if (outputDType == undefined) {
        throw new Error('Failed to determine the output dtype.');
    }
    if (axis === -1 || (axis === 0 && X.ndim === 1)) {
        // reduce to a scalar
        $S_BLOCK
    } else {
        // reduce to a tensor
        $T_BLOCK
    }
};`;

/**
 * Code block that reduces the input into a scalar (or scalar tensor when
 * keepDims = true).
 */
export const S_BLOCK_TEMPLATE =
`#ifnot NO_COMPLEX_INPUT
if (isInputComplex) {
    tmp = fComplex(X.realData, X.imagData, 0, 1, X.size);
#if OUTPUT_C_COMPLEX
#if OUTPUT_INDICES
    if (keepDims) {
        return [Tensor.scalar(tmp[0], tmp[1], outputDType, X.ndim), Tensor.scalar(tmp[2], 0, indexDType, X.ndim)];
    } else {
        return [tmp[1] === 0 ? tmp[0] : new ComplexNumber(tmp[0], tmp[1]), tmp[2]];
    }
#else
    if (keepDims) {
        return Tensor.scalar(tmp[0], tmp[1], outputDType, X.ndim);
    } else {
        return tmp[1] === 0 ? tmp[0] : new ComplexNumber(tmp[0], tmp[1]);
    }
#endif
#else
#if OUTPUT_INDICES
    if (keepDims) {
        return [Tensor.scalar(tmp[0], 0, outputDType, X.ndim), Tensor.scalar(tmp[1], 0, indexDType, X.ndim)];
    } else {
        return [tmp[0], tmp[1]];
    }
#else
    return keepDims ? Tensor.scalar(tmp, 0, outputDType, X.ndim) : tmp;
#endif
#endif
}
#endif
tmp = fReal(X.realData, 0, 1, X.size);
#if OUTPUT_R_COMPLEX
#if OUTPUT_INDICES
    if (keepDims) {
        return [Tensor.scalar(tmp[0], tmp[1], outputDType, X.ndim), Tensor.scalar(tmp[2], 0, indexDType, X.ndim)];
    } else {
        return [tmp[1] === 0 ? tmp[0] : new ComplexNumber(tmp[0], tmp[1]), tmp[2]];
    }
#else
    if (keepDims) {
        return Tensor.scalar(tmp[0], tmp[1], outputDType, X.ndim);
    } else {
        return tmp[1] === 0 ? tmp[0] : new ComplexNumber(tmp[0], tmp[1]);
    }
#endif
#else
#if OUTPUT_INDICES
    if (keepDims) {
        return [Tensor.scalar(tmp[0], 0, outputDType, X.ndim), Tensor.scalar(tmp[1], 0, indexDType, X.ndim)];
    } else {
        return [tmp[0], tmp[1]];
    }
#else
    return keepDims ? Tensor.scalar(tmp, 0, outputDType, X.ndim) : tmp;
#endif
#endif`;

/**
 * Code block that reduces the input into a tensor.
 */
export const T_BLOCK_TEMPLATE =
`var shapeX = X.shape;
var shapeY = shapeX.slice();
shapeY[axis] = 1;
var Y = Tensor.zeros(shapeY, outputDType);
var stridesX = X.strides;
var stridesY = Y.strides;
var maxLevel = X.ndim - 1;
var stride = stridesX[axis];
var n = shapeX[axis];
var reX, reY, imX, imY;
#if OUTPUT_INDICES
var Z = Tensor.zeros(shapeY, indexDType);
var reZ = Z.realData;
#endif
#if NO_COMPLEX_INPUT
$T_R_BLOCK
#else
if (isInputComplex) {
    $T_C_BLOCK
} else {
    $T_R_BLOCK
}
#endif
if (!keepDims) {
    shapeY.splice(axis, 1);
    Y.reshape(shapeY);
#if OUTPUT_INDICES
    Z.reshape(shapeY);
#endif
}
#if OUTPUT_INDICES
return [Y, Z];
#else
return Y;
#endif`;

export const T_R_BLOCK_TEMPLATE =
`reX = X.realData;
reY = Y.realData;
#if OUTPUT_R_COMPLEX
Y.ensureComplexStorage();
imY = Y.imagData;
var doReductionRICO = function (level, offsetX, offsetY) {
    var tmp;
    if (level === maxLevel) {
        for (var j = 0;j < shapeY[level];j++) {
            tmp = fReal(reX, offsetX, stride, n);
            reY[offsetY] = tmp[0];
            imY[offsetY] = tmp[1];
#if OUTPUT_INDICES
            reZ[offsetY] = tmp[2];
#endif
            offsetX++;
            offsetY++;
        }
    } else {
        for (var i = 0;i < shapeY[level];i++) {
            doReductionRICO(level + 1, offsetX, offsetY);
            offsetX += stridesX[level];
            offsetY += stridesY[level];
        }
    }
};
doReductionRICO(0, 0, 0);
#else
var doReductionRIRO = function (level, offsetX, offsetY) {
    var tmp;
    if (level === maxLevel) {
        for (var j = 0;j < shapeY[level];j++) {
            tmp = fReal(reX, offsetX, stride, n);
#if OUTPUT_INDICES
            reY[offsetY] = tmp[0];
            reZ[offsetY] = tmp[1];
#else
            reY[offsetY] = tmp;
#endif
            offsetX++;
            offsetY++;
        }
    } else {
        for (var i = 0;i < shapeY[level];i++) {
            doReductionRIRO(level + 1, offsetX, offsetY);
            offsetX += stridesX[level];
            offsetY += stridesY[level];
        }
    }
};
doReductionRIRO(0, 0, 0);
#endif
`;

export const T_C_BLOCK_TEMPLATE =
`reX = X.realData;
reY = Y.realData;
imX = X.imagData;
#if OUTPUT_C_COMPLEX
Y.ensureComplexStorage();
imY = Y.imagData;
var doReductionCICO = function (level, offsetX, offsetY) {
    var tmp;
    if (level === maxLevel) {
        for (var j = 0;j < shapeY[level];j++) {
            tmp = fComplex(reX, imX, offsetX, stride, n);
            reY[offsetY] = tmp[0];
            imY[offsetY] = tmp[1];
#if OUTPUT_INDICES
            reZ[offsetY] = tmp[2];
#endif
            offsetX++;
            offsetY++;
        }
    } else {
        for (var i = 0;i < shapeY[level];i++) {
            doReductionCICO(level + 1, offsetX, offsetY);
            offsetX += stridesX[level];
            offsetY += stridesY[level];
        }
    }
};
doReductionCICO(0, 0, 0);
#else
var doReductionCIRO = function (level, offsetX, offsetY) {
    var tmp;
    if (level === maxLevel) {
        for (var j = 0;j < shapeY[level];j++) {
            tmp = fComplex(reX, imX, offsetX, stride, n);
#if OUTPUT_INDICES
            reY[offsetY] = tmp[0];
            reZ[offsetY] = tmp[1];
#else
            reY[offsetY] = tmp;
#endif
            offsetX++;
            offsetY++;
        }
    } else {
        for (var i = 0;i < shapeY[level];i++) {
            doReductionCIRO(level + 1, offsetX, offsetY);
            offsetX += stridesX[level];
            offsetY += stridesY[level];
        }
    }
};
doReductionCIRO(0, 0, 0);
#endif
`;
