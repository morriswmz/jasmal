import { ElementWiseOpGenerator } from '../generator';
import { OpOutput, OpInput } from '../../commonTypes';
import { OutputDTypeResolver, DType } from '../../dtype';
import { Tensor } from '../../tensor';
import { DataHelper } from '../../helper/dataHelper';

export interface IPowerMathOpSet {

    /**
     * Computes the square root for each element in the input.
     */
    sqrt(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Computes the square for each element in the input.
     */
    square(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Raises `x` to the powers from `y` element by element.
     * Note: This operation accepts complex inputs and can produce complex
     *       outputs for certain real inputs (e.g., `pow(-1, 0.5)`).
     */
    pow(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Raises `x` to the powers from `y` element by element, but only allows
     * real inputs. In addition, NaN is produced when x^y is complex.
     * This operation should behave the same as `Math.pow` in JavaScript.
     */
    realpow(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

}

export class PowerMathOpSetFactory {

    public static create(generator: ElementWiseOpGenerator): IPowerMathOpSet {

        const opSqrtP = generator.makeUnaryOp({
            opR: '$reY = Math.sqrt($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opSqrtA = generator.makeUnaryOp({
            opR: 'if ($reX >= 0) { $reY = Math.sqrt($reX); } else { $imY = Math.sqrt(-$reX); $reY = 0; }',
            opC: '$tmp1 = CMath.csqrt($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat            
        });

        const opSqrt = (x: OpInput, inPlace: boolean = false): OpOutput => {
            let infoX = Tensor.analyzeOpInput(x);
            if (infoX.isComplex || infoX.re < 0 || DataHelper.anyNegative(infoX.reArr)) {
                return opSqrtA(infoX, inPlace);
            } else {
                return opSqrtP(infoX, inPlace);
            }
        };

        const opSquare = generator.makeUnaryOp({
            opR: '$reY = $reX * $reX;',
            opC: '$tmp1 = $reX * $reX - $imX * $imX; $tmp2 = $reX * $imX; $reY = $tmp1; $imY = $tmp2 + $tmp2;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat
        });

        const opPowR = generator.makeBinaryOp({
            opRR: '$reZ = Math.pow($reX, $reY);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToFloat
        });

        const opPowCC = generator.makeBinaryOp({
            opRR: '$tmp1 = CMath.cpowRR($reX, $reY); $reZ = $tmp1[0]; $imZ = $tmp1[1];',
            opRC: '$tmp1 = CMath.cpow($reX, 0, $reY, $imY); $reZ = $tmp1[0]; $imZ = $tmp1[1];',
            opCR: '$tmp1 = CMath.cpow($reX, $imX, $reY, 0); $reZ = $tmp1[0]; $imZ = $tmp1[1];',
            opCC: '$tmp1 = CMath.cpow($reX, $imX, $reY, $imY); $reZ = $tmp1[0]; $imZ = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToFloat
        });

        const opPow = (x: OpInput, y: OpInput, inPlace: boolean = false): OpOutput => {
            let infoX = Tensor.analyzeOpInput(x);
            let infoY = Tensor.analyzeOpInput(y);
            let Z: OpOutput;
            if (infoX.isComplex || infoY.isComplex) {
                Z = opPowCC(infoX, infoY, inPlace);
            } else {
                if ((infoX.re < 0 || DataHelper.anyNegative(infoX.reArr)) && infoY.originalDType === DType.FLOAT64) {
                    // When x has negative elements, it is possible to produce
                    // complex results when y is not an integer.
                    Z = opPowCC(infoX, infoY, inPlace);
                } else {
                    // x and y are both real and nonnegative
                    Z = opPowR(infoX, infoY, inPlace);
                }
            }
            if (Z instanceof Tensor && Z.hasComplexStorage()) {
                if (DataHelper.isArrayAllZeros(Z.imagData)) {
                    Z.trimImaginaryPart();
                }
            }
            return Z;
        };

        return {
            sqrt: opSqrt,
            square: opSquare,
            pow: opPow,
            realpow: opPowR
        };

    }

}
