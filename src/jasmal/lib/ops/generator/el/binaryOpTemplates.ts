import { OpInputType } from '../../../commonTypes';

/**
 * Binary element-wise operation template hierarchy:
 * + BIN_EL_OP_TEMPLATE
 * | + TT_BLOCK_TEMPLATE
 * | | + TT_NORMAL_BLOCK_TEMPLATE
 * | | + TT_BROADCAST_BLOCK_TEMPLATE
 * | | | + TT_BROADCAST_SUB_BLOCK_TEMPLATE
 * | + TS_BLOCK_TEMPLATE
 * | + ST_BLOCK_TEMPLATE
 * | + SS_BLOCK_TEMPLATE
 * 
 * Although the (tensor, scalar) cases can be handled by our broadcasting
 * implementation in $TTBlock, they are handled separately for speed. For these
 * two cases, a simple for loop is sufficient. 
 */

/**
 * Top-level template for the binary operation, requiring the following blocks:
 *  $TTBlock - Block of codes that handles op(Tensor, Tensor)
 *  $TSBlock - Block of codes that handles op(Tensor, Scalar)
 *  $STBlock - Block of codes that handles op(Scalar, Tensor)
 *  $SSBlock - Block of codes that handles op(Scalar, Scalar)
 * Note that not all operations are commutative so all four blocks are required.
 * In-place operations are allowed in the following cases:
 *  x is Tensor and broadcasting does not require changing the shape of x.
 * Note: internally, we allow x and y to be the results returned by
 * Tensor.analyzeOpInput() to avoid reevaluating Tensor.analyzeOpInput(). 
 */
export const BIN_EL_OP_TEMPLATE =
`'use strict';
$Dependencies
$InlineFunctions
return function (x, y, inPlace) {
    // process inputs
    inPlace = inPlace || false;
    // init common variables
    var infoX, infoY;
    if (x['originalType'] != undefined) {
        infoX = x;
        x = infoX.originalInput;
    } else {
        infoX = Tensor.analyzeOpInput(x);
    }
    if (y['originalType'] != undefined) {
        infoY = y;
        y = infoY.originalInput;
    } else {
        infoY = Tensor.analyzeOpInput(y);
    }
    if (infoX.originalType === ${OpInputType.Unknown} || infoY.originalType === ${OpInputType.Unknown}) {
        throw new Error('Unsupported input type.');
    }
    #if NO_IN_PLACE
    if (inPlace) {
        throw new Error('In-place operation is not supported.');
    }
    #else
    if (inPlace && infoX.originalType !== ${OpInputType.Tensor}) {
        throw new Error('In-place operation is not allowed when the first operand is not a tensor.');
    }
    #endif
    #if NO_COMPLEX_INPUT
    if (infoX.isComplex) {
        throw new Error('Complex input is not allowed.');
    }
    if (infoY.isComplex) {
        throw new Error('Complex input is not allowed.');
    }
    #endif
    var isXScalar = infoX.isInputScalar;
    var isYScalar = infoY.isInputScalar;
    var shouldOutputTensor = !isXScalar || !isYScalar;
    var dtypeX = infoX.originalDType, dtypeY = infoY.originalDType, dtypeZ;
    var reXScalar = infoX.re, imXScalar = infoX.im, reYScalar = infoY.re, imYScalar = infoY.im;
    var reZScalar = 0, imZScalar = 0;
    var reX = infoX.reArr, imX = infoX.imArr, reY = infoY.reArr, imY = infoY.imArr;
    var reZ, imZ, tmp1, tmp2, tmp3, tmp4, z, s;
    var i = 0;
    // check dtype
    dtypeZ = outputDTypeResolver(dtypeX, infoX.isComplex, dtypeY, infoY.isComplex);
    if (dtypeZ == undefined) {
        throw new Error('Operation between ' + DTypeHelper.dTypeToString(dtypeX) + ' and ' 
            + DTypeHelper.dTypeToString(dtypeY) + ' is not available.');
    }
    // main procedure
    if (isXScalar) {
        if (isYScalar) {
            
            $SSBlock
        } else {
            $STBlock
        }
    } else {
#ifnot NO_IN_PLACE
        if (inPlace && DTypeHelper.isWiderType(dtypeX, dtypeZ)) {
            throw new Error('Cannot downcast from ' + DTypeHelper.dTypeToString(dtypeY) + ' to ' +
                DTypeHelper.dTypeToString(dtypeX) + ' when performing in-place operation.');
        }
#endif
        if (isYScalar) {
            $TSBlock
        } else {
            $TTBlock
        }
    }
    return z;
}`;

/**
 * Template for the block of codes that handles op(Scalar, Scalar). This
 * template requires the following four blocks:
 *  $RRBlock - Block of codes that handles the real-real operation.
 *  $RCBlock - Block of codes that handles the real-complex operation.
 *  $CRBlock - Block of codes that handles the complex-real operation.
 *  $CCBlock - Block of codes that handles the complex-complex operation.
 * Note that not all operations are commutative so all four blocks are required.
 * These four blocks will be compiled from the specified CoreOpTemplate.
 */
export const SS_BLOCK_TEMPLATE =
`// reX, imX, reY, imY have already been set when processing inputs
#if NO_COMPLEX_INPUT
$RRBlock
#else
if (infoX.isComplex) {
    if (infoY.isComplex) {
        $CCBlock
    } else {
        $CRBlock
    }
} else {
    if (infoY.isComplex) {
        $RCBlock
    } else {
        $RRBlock
    }
}
#endif
z = imZScalar === 0 ? reZScalar : new ComplexNumber(reZScalar, imZScalar);`;

/**
 * Template for the block of codes that handles op(Scalar, Tensor). This
 * template requires the following four blocks:
 *  $RRBlock - Block of codes that handles the real-real operation.
 *  $RCBlock - Block of codes that handles the real-complex operation.
 *  $CRBlock - Block of codes that handles the complex-real operation.
 *  $CCBlock - Block of codes that handles the complex-complex operation.
 * Note that not all operations are commutative so all four blocks are required.
 * These four blocks will be compiled from the specified CoreOpTemplate.
 */
export const ST_BLOCK_TEMPLATE =
`// reX, imX, reY, imY have already been set when processing inputs
#if NO_IN_PLACE
#else
if (inPlace) {
    throw new Error('In-place operation cannot be performed because the output shape is different from that of the first operand.')
}
#endif
z = Tensor.zeros(infoY.originalShape, dtypeZ);
reZ = z.realData;
#if NO_COMPLEX_INPUT
#if OUTPUT_RR_COMPLEX
z.ensureComplexStorage();
imZ = z.imagData;
#endif
for (i = 0;i < reZ.length;i++) {
    // reX, reY[] -> reZ[], imZ[]?
    $RRBlock
}
#else
if (infoX.isComplex) {
    if (infoY.isComplex) {  
#if OUTPUT_CC_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        for (i = 0;i < reZ.length;i++) {
            // reX, imX, reY[], imY[] -> reZ[], imZ[]?
            $CCBlock
        }
    } else {
#if OUTPUT_CR_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        for (i = 0;i < reZ.length;i++) {
            // reX, imX, reY[] -> reZ[], imZ[]?
            $CRBlock
        }
    }
} else {
    if (infoY.isComplex) {
#if OUTPUT_RC_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        for (i = 0;i < reZ.length;i++) {
            // reX, reY[], imY[] -> reZ[], imZ[]?
            $RCBlock
        }
    } else {
#if OUTPUT_RR_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        for (i = 0;i < reZ.length;i++) {
            // reX, reY[] -> reZ[], imZ[]?
            $RRBlock
        }
    }
}
#endif`;

/**
 * Template for the block of codes that handles op(Tensor, Scalar). This
 * template requires the following four blocks:
 *  $RRBlock - Block of codes that handles the real-real operation.
 *  $RCBlock - Block of codes that handles the real-complex operation.
 *  $CRBlock - Block of codes that handles the complex-real operation.
 *  $CCBlock - Block of codes that handles the complex-complex operation.
 * Note that not all operations are commutative so all four blocks are required.
 * These four blocks will be compiled from the specified CoreOpTemplate.
 */
export const TS_BLOCK_TEMPLATE =
`// reX, imX, reY, imY have already been set when processing inputs
#if NO_IN_PLACE
z = Tensor.zeros(infoX.originalShape, dtypeZ);
#else
if (inPlace) {
    z = x;
    z.ensureUnsharedLocalStorage();
} else {
    z = Tensor.zeros(infoX.originalShape, dtypeZ);
}
#endif
reZ = z.realData;
#if NO_COMPLEX_INPUT
#if OUTPUT_RR_COMPLEX
z.ensureComplexStorage();
imZ = z.imagData;
#endif
for (i = 0;i < reZ.length;i++) {
    // reX[], reY -> reZ[], imZ[]?
    $RRBlock
}
#else
if (infoY.isComplex) {
    if (infoX.isComplex) {
#if OUTPUT_CC_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        for (i = 0;i < reZ.length;i++) {
            // reX[], imX[], reY, imY -> reZ[], imZ[]?
            $CCBlock
        }
#if !NO_IN_PLACE && !OUTPUT_CC_COMPLEX
        if (inPlace) {
            for (i = 0;i < imX.length;i++) {
                imX[i] = 0;
            }
        }
#endif
    } else {
#if OUTPUT_RC_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        for (i = 0;i < reZ.length;i++) {
            // reX[], reY, imY -> reZ[], imZ[]?
            $RCBlock
        }
    }
} else {
    if (infoX.isComplex) {
#if OUTPUT_CR_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        for (i = 0;i < reZ.length;i++) {
            // reX[], imX[], reY -> reZ[], imZ[]?
            $CRBlock
        }
#if !NO_IN_PLACE && !OUTPUT_CR_COMPLEX
        if (inPlace) {
            for (i = 0;i < imX.length;i++) {
                imX[i] = 0;
            }
        }
#endif
    } else {
#if OUTPUT_RR_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        for (i = 0;i < reZ.length;i++) {
            // reX[], reY -> reZ[], imZ[]?
            $RRBlock
        }
    }
}
#endif`;

/**
 * Template for the block of codes that handles op(Tensor, Tensor). This
 * template requires the following two blocks:
 *  $TTNormalBlock - Block codes that handles op(Tensor, Tensor) without
 *      broadcasting (in this case the two input tensors share the same shape.)
 *  $TTBroadcastBlock - Block codes that handles op(Tensor, Tensor) with
 *      broadcasting (in this case the two input tensors have compatible
 *      shapes.)
 */
export const TT_BLOCK_TEMPLATE =
`var results, shapeX, shapeY, shapeZ;
results = ShapeHelper.checkBroadcastingCompatibility(infoX.originalShape, infoY.originalShape);
shapeX = results.shapeX;
shapeY = results.shapeY;
shapeZ = results.shapeZ;
if (results.exact) {
#if NO_IN_PLACE
    z = Tensor.zeros(shapeZ, dtypeZ);
#else
    if (inPlace) {
        z = x;
        z.ensureUnsharedLocalStorage();
    } else {
        z = Tensor.zeros(shapeZ, dtypeZ);
    }
#endif
    $TTNormalBlock
} else {
#if NO_IN_PLACE
    z = Tensor.zeros(shapeZ, dtypeZ);
#else
    if (inPlace) {
        if (!ShapeHelper.compareShape(shapeX, shapeZ)) {
            throw new Error('Cannot perform in-place operations when the output shape different from that of the first operand.');
        }
        z = x;
        z.ensureUnsharedLocalStorage();
    } else {
        z = Tensor.zeros(shapeZ, dtypeZ);
    }
#endif
    $TTBroadcastBlock
}`;

/**
 * Template for the block of codes that handles op(Tensor, Tensor) without
 * broadcasting. This template requires the following four blocks:
 *  $RRBlock - Block of codes that handles the real-real operation.
 *  $RCBlock - Block of codes that handles the real-complex operation.
 *  $CRBlock - Block of codes that handles the complex-real operation.
 *  $CCBlock - Block of codes that handles the complex-complex operation.
 * Note that not all operations are commutative so all four blocks are required.
 * These four blocks will be compiled from the specified CoreOpTemplate.
 */
export const TT_NORMAL_BLOCK_TEMPLATE =
`reZ = z.realData;
#if NO_COMPLEX_INPUT
#if OUTPUT_RR_COMPLEX
z.ensureComplexStorage();
imZ = z.imagData;
#endif
for (i = 0;i < reZ.length;i++) {
    // reX[], reY[] -> reZ[], imZ[]?
    $RRBlock
}
#else
if (infoX.isComplex) {
    reZ = z.realData;
    if (infoY.isComplex) {
#if OUTPUT_CC_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        for (i = 0;i < reZ.length;i++) {
            // reX[], imX[], reY[], imY[] -> reZ[], imZ[]?
            $CCBlock
        }
    } else {
#if OUTPUT_CR_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        reY = y.realData;
        for (i = 0;i < reZ.length;i++) {
            // reX[], imX[], reY[] -> reZ[], imZ[]?
            $CRBlock
        }
    }
#if !NO_IN_PLACE && (!OUTPUT_CC_COMPLEX || !OUTPUT_CR_COMPLEX)
    if (inPlace) {
        for (i = 0;i < imX.length;i++) {
            imX[i] = 0;
        }
    }
#endif
} else {
    if (infoY.isComplex) {
#if OUTPUT_RC_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        for (i = 0;i < reZ.length;i++) {
            // reX[], reY[], imY[] -> reZ[], imZ[]?
            $RCBlock
        }
    } else {
#if OUTPUT_RR_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        for (i = 0;i < reZ.length;i++) {
            // reX[], reY[] -> reZ[], imZ[]?
            $RRBlock
        }
    }
}
#endif`;

/**
 * Template for the block of codes that handles op(Tensor, Tensor) with
 * broadcasting. This template requires the following four blocks:
 *  $RRBlock - Block of codes that handles the real-real operation.
 *  $RCBlock - Block of codes that handles the real-complex operation.
 *  $CRBlock - Block of codes that handles the complex-real operation.
 *  $CCBlock - Block of codes that handles the complex-complex operation.
 * Note that not all operations are commutative so all four blocks are required.
 * These four blocks will be compiled from TT_BROADCAST_SUB_BLOCK_TEMPLATE.
 */
export const TT_BROADCAST_BLOCK_TEMPLATE =
`// Because the number of dimensions are not fixed, we cannot use for loops here directly.
// We will use recursion here.
var stridesX = ShapeHelper.computeStrides(shapeX);
var stridesY = ShapeHelper.computeStrides(shapeY);
var stridesZ = ShapeHelper.computeStrides(shapeZ);
reZ = z.realData;
#if NO_COMPLEX_INPUT
#if OUTPUT_RR_COMPLEX
z.ensureComplexStorage();
imZ = z.imagData;
#endif
$RRBlock
#else
if (infoX.isComplex) {
    if (infoY.isComplex) {
#if OUTPUT_CC_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        $CCBlock
    } else {
#if OUTPUT_CR_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        $CRBlock
    }
#if !NO_IN_PLACE && (!OUTPUT_CC_COMPLEX || !OUTPUT_CR_COMPLEX)
    if (inPlace) {
        for (i = 0;i < imX.length;i++) {
            imX[i] = 0;
        }
    }
#endif
} else {
    if (infoY.isComplex) {
#if OUTPUT_RC_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        $RCBlock
    } else {
#if OUTPUT_RR_COMPLEX
        z.ensureComplexStorage();
        imZ = z.imagData;
#endif
        $RRBlock
    }
}
#endif`;

/**
 * Template for the generate the four blocks in TT_BROADCAST_BLOCK_TEMPLATE.
 * This templates requires the following three blocks:
 *  $OpFixX - Block of codes implementing the core operation with the index of
 *      x fixed.
 *  $OpFixY - Block of codes implementing the core operation with the index of
 *      y fixed.
 *  $OpNormal - Block of codes implementing the core operation.
 * These three blocks will be compiled from the specified CoreOpTemplate.
 */
export const TT_BROADCAST_SUB_BLOCK_TEMPLATE =
`var applyOp = function (level, offsetX, offsetY, offsetZ) {
    var i = 0;
    if (level === shapeZ.length - 1) {
        // last level
        if (shapeX[level] === 1) {
            // z[offsetZ + i] <- op(x[offsetX], y[offset + i])
            for (i = 0;i < shapeZ[level];i++) {
                $OpFixX
            }
        } else if (shapeY[level] === 1) {
            // z[offsetZ + i] <- op(x[offsetX + i], y[offset])
            for (i = 0;i < shapeZ[level];i++) {
                $OpFixY
            }
        } else {
            // z[offsetZ + i] <- op(x[offsetX + i], y[offset + i])
            for (i = 0;i < shapeZ[level];i++) {
                $OpNormal
            }
        }
    } else {
        if (shapeX[level] === 1) {
            // fix index[level] = 1 for x
            for (i = 0;i < shapeZ[level];i++) {
                applyOp(level + 1, offsetX, offsetY, offsetZ);
                offsetY += stridesY[level];
                offsetZ += stridesZ[level];
            }
        } else if (shapeY[level] === 1) {
            // fix index[level] = 1 for y
            for (i = 0;i < shapeZ[level];i++) {
                applyOp(level + 1, offsetX, offsetY, offsetZ);
                offsetX += stridesX[level];
                offsetZ += stridesZ[level];
            }
        } else {
            for (i = 0;i < shapeZ[level];i++) {
                applyOp(level + 1, offsetX, offsetY, offsetZ);
                offsetX += stridesX[level];
                offsetY += stridesY[level];
                offsetZ += stridesZ[level];
            }
        }
    }
}
applyOp(0, 0, 0, 0);`;
