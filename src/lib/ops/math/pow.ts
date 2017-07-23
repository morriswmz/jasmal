import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { OpOutput, OpInput } from '../../commonTypes';
import { OutputDTypeResolver } from '../../dtype';
import { CMathHelper } from '../../helper/mathHelper';
import { Tensor } from '../../tensor';
import { DataHelper } from '../../helper/dataHelper';

export interface IPowerMathOpSet {

    sqrt(x: OpInput, inPlace?: boolean): OpOutput;

    pow2(x: OpInput, y: OpInput, inPlace?: boolean): OpOutput;

}

export class PowerMathOpSetFactory {

    public static create(compiler: TensorElementWiseOpCompiler): IPowerMathOpSet {

        // TODO: negative, frac
        const opPow = compiler.makeOneParamUnaryOp({
            opR: '$reY = Math.pow($reX, $param);',
            opC: '$tmp1 = $param * Math.atan($imX / $reX);\n' +
                 'if (Math.abs($reX) < Math.abs($imX)) {\n' +
                 '    $tmp2 = $reX / $imX;\n' +
                 '    $tmp3 = $imX * Math.sqrt(1 + $tmp2 * $tmp2);\n' +
                 '} else {\n' +
                 '    $tmp2 = $imX / $reX;\n' +
                 '    $tmp3 = $reX * Math.sqrt(1 + $tmp2 * $tmp2);\n' +
                 '}\n' +
                 '$tmp4 = Math.pow($tmp3, $param);\n' +
                 '$reY = $tmp4 * Math.cos($tmp1); $imY = $tmp4 * Math.sin($tmp1);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });


        const opSqrtP = compiler.makeUnaryOp({
            opR: '$reY = Math.sqrt($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opSqrtA = compiler.makeUnaryOp({
            opR: 'if ($reX >= 0) { $reY = Math.sqrt($reX); } else { $reY = 0; $imY = Math.sqrt(-$reX); }',
            opC: '$tmp1 = csqrt($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            // sqrt(z) => [sqrt(0.5*(Re(z) + |z|)), 0.5*Im(z) / sqrt(0.5*(Re(z) + |z|))]
            inlineFunctions: {
                'csqrt': CMathHelper.csqrt
            }
        });

        const opSqrt = (x: OpInput, inPlace: boolean = false): OpOutput => {
            let infoX = Tensor.analyzeOpInput(x);
            if (infoX.isComplex || infoX.re < 0 || DataHelper.anyNegative(infoX.reArr)) {
                return opSqrtA(infoX, inPlace);
            } else {
                return opSqrtP(infoX, inPlace);
            }
        };

        const opPow2 = (x: OpInput, y: OpInput, inPlace: boolean = false): OpOutput => {
            throw new Error('Not implemented');
        };

        return {
            sqrt: opSqrt,
            pow2: opPow2
        };

    }

}