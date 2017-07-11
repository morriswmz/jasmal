import { IMathOpProvider } from '../definition';
import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { DType, OutputDTypeResolver } from '../../dtype';
import { Tensor } from '../../tensor';
import { OpInput, OpOutput, OpInputType } from '../../commonTypes';
import { MathHelper } from '../../helper/mathHelper';

export class MathOpProviderFactory {
    public static create(): IMathOpProvider {
        
        const compiler = TensorElementWiseOpCompiler.GetInstance();

        const notImplemented = () => {
            throw new Error('Not implemented.');
        };

        const opAbs = compiler.makeUnaryOp({
            opR: '$reY = Math.abs($reX);',
            opC: '$reY = length2($reX, $imX);',
        }, {
            // custom rule here, only convert to float when the input is complex
            outputDTypeResolver: (t, isComplex) => isComplex ? DType.FLOAT64 : t,
            inlineFunctions: {
                'length2': MathHelper.length2
            }
        });

        const opSign = compiler.makeUnaryOp({
            opR: '$reY = $reX > 0 ? 1 : ($reX < 0 ? -1 : ($reX === 0 ? 0 : NaN));',
            opC: '$reY = $reX > 0 ? 1 : ($reX < 0 ? -1 : ($reX === 0 ? 0 : NaN));\n' + 
                 '$imY = $imX > 0 ? 1 : ($imX < 0 ? -1 : ($imX === 0 ? 0 : NaN))'
        });

        const opMin2 = compiler.makeBinaryOp({
            opRR: '$reZ = Math.min($reX, $reY);'
        });

        const opMax2 = compiler.makeBinaryOp({
            opRR: '$reZ = Math.max($reX, $reY);'
        });

        const opConj = compiler.makeUnaryOp({
            opR: '$reY = $reX;',
            opC: '$reY = $reX; $imY = -$imX;'
        });

        const opAngle = compiler.makeUnaryOp({
            opR: '$reY = 0;',
            opC: '$reY = Math.atan2($imX, $reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opExp = compiler.makeUnaryOp({
            opR: '$reY = Math.exp($reX);',
            opC: '$tmp1 = Math.exp($reX); $reY = $tmp1 * Math.cos($imX); $imY = $tmp1 * Math.sin($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

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

        const opSin = compiler.makeUnaryOp({
            opR: '$reY = Math.sin($reX);',
            opC: '$tmp1 = Math.exp($imX); $tmp2 = Math.exp(-$imX); $tmp3 = $reX;' +
                '$reY = 0.5 * Math.sin($tmp3) * ($tmp1 + $tmp2);' +
                '$imY = 0.5 * Math.cos($tmp3) * ($tmp1 - $tmp2);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opCos = compiler.makeUnaryOp({
            opR: '$reY = Math.cos($reX);',
            opC: '$tmp1 = Math.exp($imX); $tmp2 = Math.exp(-$imX); $tmp3 = $reX;' +
                '$reY = 0.5 * Math.cos($tmp3) * ($tmp1 + $tmp2);' +
                '$imY = 0.5 * Math.sin($tmp3) * ($tmp1 - $tmp2);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opTan = compiler.makeUnaryOp({
            opR: '$reY = Math.tan($reX);'
            //           sin(2*Re(z)) + j sinh(2*Im(z))
            // tan(z) = --------------------------------
            //            cosh(2*Im(z)) + cos(2*Re(z))
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opCot = compiler.makeUnaryOp({
            opR: '$reY = 1.0 / Math.tan($reX);'
            //           sin(2*Re(z)) - j sinh(2*Im(z))
            // cot(z) = --------------------------------
            //            cosh(2*Im(z)) - cos(2*Re(z))
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opAsin = compiler.makeUnaryOp({
            opR: '$reY = Math.asin($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opAcos = compiler.makeUnaryOp({
            opR: '$reY = Math.acos($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opAtan = compiler.makeUnaryOp({
            opR: '$reY = Math.atan($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opSqrtP = compiler.makeUnaryOp({
            opR: '$reY = Math.sqrt($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        // TODO: Infinity and NaN handling
        const opSqrtA = compiler.makeUnaryOp({
            opR: 'if ($reX >= 0) { $reY = Math.sqrt($reX); } else { $reY = 0; $imY = Math.sqrt(-$reX); }',
            // sqrt(z) => [sqrt(0.5*(Re(z) + |z|)), 0.5*Im(z) / sqrt(0.5*(Re(z) + |z|))]
            opC: 'if ($imX === 0) {\n' +
                 '    if ($reX >= 0) { $reY = Math.sqrt($reX); } else { $reY = 0; $imY = Math.sqrt(-$reX); }\n' +
                 '} else {\n' +
                 '    if (Math.abs($reX) < Math.abs($imX)) {\n' +
                 '        $tmp1 = $reX / $imX;\n' +
                 '        $tmp2 = Math.sqrt(0.5 * ($reX + $imX * Math.sqrt(1 + $tmp1 * $tmp1)));\n' +
                 '        $reY = $tmp2; $imY = 0.5 * $imX / $tmp2;\n' +
                 '    } else {\n' +
                 '        if ($reX === 0) {\n' +
                 '            $reY = 0; $imY = 0;\n' +
                 '        } else {\n' +
                 '            $tmp1 = $imX / $reX;\n' +
                 '            $tmp2 = Math.sqrt(0.5 * $reX * (1 + Math.sqrt(1 + $tmp1 * $tmp1)));\n' +
                 '            $reY = $tmp2; $imY = 0.5 * $imX / $tmp2;\n' +
                 '        }\n' +
                 '    }\n' +
                 '}\n'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opFloor = compiler.makeUnaryOp({
            opR: '$reY = Math.floor($reX);',
            opC: '$reY = Math.floor($reX); $imY = Math.floor($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat
        });

        const opCeil = compiler.makeUnaryOp({
            opR: '$reY = Math.ceil($reX);',
            opC: '$reY = Math.ceil($reX); $imY = Math.ceil($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat
        });

        const opRound = compiler.makeUnaryOp({
            opR: '$reY = Math.round($reX);',
            opC: '$reY = Math.round($reX); $imY = Math.round($imX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat
        });

        const opRad2Deg = compiler.makeUnaryOp({
            opR: '$reY = 180 / Math.PI * $reX;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opDeg2Rad = compiler.makeUnaryOp({
            opR: '$reY = Math.PI / 180 * $reX;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const provider: IMathOpProvider = {
            abs: opAbs,
            sign: opSign,
            min2: opMin2,
            max2: opMax2,
            conj: opConj,
            angle: opAngle,
            sin: opSin,
            cos: opCos,
            tan: opTan,
            cot: opCot,
            sinh: notImplemented,
            cosh: notImplemented,
            tanh: notImplemented,
            asin: opAsin,
            acos: opAcos,
            atan: opAtan,
            sqrt: opSqrtA,
            exp: opExp,
            pow2: notImplemented,
            log: notImplemented,
            floor: opFloor,
            ceil: opCeil,
            round: opRound,
            rad2deg: opRad2Deg,
            deg2rad: opDeg2Rad
        };

        return provider;
    }
}