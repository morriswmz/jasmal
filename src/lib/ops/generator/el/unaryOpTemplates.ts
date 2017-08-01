import { OpInputType } from '../../../commonTypes';

/**
 * Note: internally, we allow x to be the result returned by
 * Tensor.analyzeOpInput() to avoid reevaluating Tensor.analyzeOpInput(). 
 */

export const UNARY_OP_TEMPLATE =
`'use strict';
var CMath = __dep__.CMath;
$InlineFunctions
#if HAS_PARAM
return function(x, param, inPlace) {
#else
return function(x, inPlace) {
#endif
    inPlace = inPlace || false;
    // we accept OpInputInfo as an input 
    var infoX = x['originalType'] != undefined ? x : __dep__.Tensor.analyzeOpInput(x);
    if (infoX.originalType === ${OpInputType.Unknown}) {
        throw new Error('Unsupported input type.');
    }
    #if NO_IN_PLACE
    if (inPlace) {
        throw new Error('In-place operation is not supported.');
    }
    #else
    if (inPlace && infoX.originalType !== ${OpInputType.Tensor}) {
        throw new Error('Cannot perform in-place operations when the operand is not a tensor.');
    }
    #endif
    #if NO_COMPLEX_INPUT
    if (infoX.isComplex) {
        throw new Error('Complex input is not supported.');
    }
    #endif
    var dtypeX = infoX.originalDType, dtypeY;
    var reX = infoX.reArr, imX = infoX.imArr, reY, imY, tmp1, tmp2, tmp3, tmp4, y;
    var i = 0;
    dtypeY = __dep__.determineOutputType(dtypeX, infoX.isComplex);
    if (dtypeY == undefined) {
        throw new Error('The operation on ' + __dep__.dTypeToString(dtypeX) + ' is not available.');
    }
    if (infoX.isInputScalar) {
        var reXScalar = infoX.re, imXScalar = infoX.im;
        var reYScalar = 0, imYScalar = 0;
        $SBlock
        return imYScalar === 0 ? reYScalar : new __dep__.ComplexNumber(reYScalar, imYScalar);
    } else {
        if (inPlace) {
            if (__dep__.isWiderType(dtypeX, dtypeY)) {
                throw new Error('Cannot perform in-place operations for data type ' + __dep__.dTypeToString(dtypeX) + '.');
            }
            y = x;
            y.ensureUnsharedLocalStorage();
        } else {
            y = __dep__.Tensor.zeros(infoX.originalShape, dtypeY);
        }
        $TBlock
        return y;
    }
}`;

export const S_BLOCK_TEMPLATE =
`#if NO_COMPLEX_INPUT
$RBlock
#else
if (infoX.isComplex) {
    $CBlock
} else {
    $RBlock
}
#endif`;

/**
 * If input x is real, output y is real. If input x is complex, output y is
 * complex.
 */
export const T_BLOCK_TEMPLATE = 
`#if NO_COMPLEX_INPUT
reY = y.realData;
#if OUTPUT_R_COMPLEX
y.ensureComplexStorage();
imY = y.imagData;
#endif
for (i = 0;i < reY.length;i++) {
    $RBlock
}
#else
if (infoX.isComplex) {
    reY = y.realData;
#if OUTPUT_C_COMPLEX
    y.ensureComplexStorage();
    imY = y.imagData;
#endif
    for (i = 0;i < reY.length;i++) {
        $CBlock
    }
} else {
    reY = y.realData;
#if OUTPUT_R_COMPLEX
    y.ensureComplexStorage();
    imY = y.imagData;
#endif
    for (i = 0;i < reY.length;i++) {
        $RBlock
    }
}
#endif`;
