import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { OpOutput, OpInput } from '../../commonTypes';
import { OutputDTypeResolver } from '../../dtype';
import { NOT_IMPLEMENTED } from '../../constant';
import { CMathHelper } from '../../helper/mathHelper';

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
            opC: '$tmp1 = csin($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {
                'csin': CMathHelper.csin
            }
        });

        const opCos = compiler.makeUnaryOp({
            opR: '$reY = Math.cos($reX);',
            opC: '$tmp1 = ccos($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {
                'ccos': CMathHelper.ccos
            }
        });

        const opTan = compiler.makeUnaryOp({
            opR: '$reY = Math.tan($reX);',
            opC: '$tmp1 = ctan($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {
                'ctan': CMathHelper.ctan
            }
        });

        const opCot = compiler.makeUnaryOp({
            opR: '$tmp1 = Math.tan($reX); $reY = $tmp1 === 0.0 ? NaN : 1.0 / $tmp1;',
            opC: '$tmp2 = ccot($reX, $imX); $reY = $tmp2[0]; $imY = $tmp2[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {
                'ccot': CMathHelper.ccot
            }
        });

        // TODO: implement complex version of inverse trig functions

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

        const opAcot = compiler.makeUnaryOp({
            opR: '$reY = Math.PI * 0.5 - Math.atan($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opSinh = compiler.makeUnaryOp({
            opR: '$reY = 0.5*(Math.exp($reX) - Math.exp(-$reX));',
            opC: '$tmp1 = csinh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {
                'csinh': CMathHelper.csinh
            }
        });

        const opCosh = compiler.makeUnaryOp({
            opR: '$reY = 0.5*(Math.exp($reX) + Math.exp(-$reX));',
            opC: '$tmp1 = ccosh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {
                'ccosh': CMathHelper.ccosh
            }
        });

        const opTanh = compiler.makeUnaryOp({
            opR: '$tmp1 = Math.exp($reX); $tmp2 = Math.exp(-$reX); $reY = ($tmp1 - $tmp2) / ($tmp1 + $tmp2);',
            opC: '$tmp3 = ctanh($reX, $imX); $reY = $tmp3[0]; $imY = $tmp3[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {
                'ctanh': CMathHelper.ctanh
            }
        });

        const opCoth = compiler.makeUnaryOp({
            opR: '$tmp1 = Math.exp($reX); $tmp2 = Math.exp(-$reX); $reY = ($tmp1 + $tmp2) / ($tmp1 - $tmp2);',
            opC: '$tmp3 = ccoth($reX, $imX); $reY = $tmp3[0]; $imY = $tmp3[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            inlineFunctions: {
                'ccoth': CMathHelper.ccoth
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
            acot: opAcot,
            sinh: opSin,
            cosh: opCosh,
            tanh: opTanh,
            coth: opCoth
        };

    }

}