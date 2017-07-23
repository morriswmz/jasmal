import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { OpOutput, OpInput } from '../../commonTypes';
import { OutputDTypeResolver } from '../../dtype';
import { NOT_IMPLEMENTED } from '../../constant';

export interface ITrigMathOpSet {

    sin(x: OpInput, inPlace?: boolean): OpOutput;

    cos(x: OpInput, inPlace?: boolean): OpOutput;

    tan(x: OpInput, inPlace?: boolean): OpOutput;

    cot(x: OpInput, inPlace?: boolean): OpOutput;

    sinh(x: OpInput, inPlace?: boolean): OpOutput;

    cosh(x: OpInput, inPlace?: boolean): OpOutput;

    tanh(x: OpInput, inPlace?: boolean): OpOutput;
    
    coth(x: OpInput, inPlace?: boolean): OpOutput;

    asin(x: OpInput, inPlace?: boolean): OpOutput;

    acos(x: OpInput, inPlace?: boolean): OpOutput;

    atan(x: OpInput, inPlace?: boolean): OpOutput;

    acot(x: OpInput, inPlace?: boolean): OpOutput;

}

export class TrigMathOpSetFactory {

    public static create(compiler: TensorElementWiseOpCompiler): ITrigMathOpSet {

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

        const opSinh = compiler.makeUnaryOp({
            opR: '$reY = 0.5*(Math.exp($reX) - Math.exp(-$reX));',
            opC: '$tmp1 = csinh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {
                'csinh': (re: number, im: number): [number, number] => {
                    let s = Math.sin(im), c = Math.cos(im);
                    let rp = Math.exp(re), rn = Math.exp(-re);
                    return [0.5 * (rp - rn) * c, 0.5 * (rp + rn) * s];
                }
            }
        });

        const opCosh = compiler.makeUnaryOp({
            opR: '$reY = 0.5*(Math.exp($reX) + Math.exp(-$reX));',
            opC: '$tmp1 = ccosh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {
                'ccosh': (re: number, im: number): [number, number] => {
                    let s = Math.sin(im), c = Math.cos(im);
                    let rp = Math.exp(re), rn = Math.exp(-re);
                    return [0.5 * (rp + rn) * c, 0.5 * (rp - rn) * s];
                }
            }
        });

        const opTanh = compiler.makeUnaryOp({
            opR: '',
            opC: ''
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {

            }
        });

        return {
            sin: opSin,
            cos: opCos,
            tan: opTan,
            cot: opCot,
            asin: opAsin,
            acos: opAcos,
            atan: opAtan,
            acot: NOT_IMPLEMENTED,
            sinh: opSin,
            cosh: opCosh,
            tanh: NOT_IMPLEMENTED,
            coth: NOT_IMPLEMENTED
        };

    }

}