import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { OpOutput, OpInput } from '../../commonTypes';
import { OutputDTypeResolver, DType } from '../../dtype';
import { CMath } from '../../complexNumber';
import { Tensor } from '../../tensor';
import { DataHelper } from '../../helper/dataHelper';

export interface IPowerMathOpSet {

    sqrt(x: OpInput, inPlace?: boolean): OpOutput;

    pow2(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

}

export class PowerMathOpSetFactory {

    public static create(compiler: TensorElementWiseOpCompiler): IPowerMathOpSet {

        const opSqrtP = compiler.makeUnaryOp({
            opR: '$reY = Math.sqrt($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opSqrtA = compiler.makeUnaryOp({
            opR: 'if ($reX >= 0) { $reY = Math.sqrt($reX); } else { $reY = 0; $imY = Math.sqrt(-$reX); }',
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

        const opPowR = compiler.makeBinaryOp({
            opRR: '$reZ = Math.pow($reX, $reY);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToFloat
        });

        const opPowCC = compiler.makeBinaryOp({
            opRR: '$tmp1 = CMath.cpowRR($reX, $reY); $reZ = $tmp1[0]; $imZ = $tmp1[1];',
            opRC: '$tmp1 = CMath.cpow($reX, 0, $reY, $imY); $reZ = $tmp1[0]; $imZ = $tmp1[1];',
            opCR: '$tmp1 = CMath.cpow($reX, $imX, $reY, 0); $reZ = $tmp1[0]; $imZ = $tmp1[1];',
            opCC: '$tmp1 = CMath.cpow($reX, $imX, $reY, $imY); $reZ = $tmp1[0]; $imZ = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bToFloat
        });

        const opPow2 = (x: OpInput, y: OpInput, inPlace: boolean = false): OpOutput => {
            let infoX = Tensor.analyzeOpInput(x);
            let infoY = Tensor.analyzeOpInput(y);
            let Z: OpOutput;
            if (infoX.isComplex || infoY.isComplex) {
                Z = opPowCC(infoX, infoY);
            } else {
                if ((infoX.re < 0 || DataHelper.anyNegative(infoX.reArr)) && infoY.originalDType === DType.FLOAT64) {
                    // When x has negative elements, it is possible to produce
                    // complex results when y is not an integer.
                    Z = opPowCC(infoX, infoY);
                } else {
                    // x and y are both real and nonnegative
                    Z = opPowR(infoX, infoY);
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
            pow2: opPow2
        };

    }

}