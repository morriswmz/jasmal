import { ElementWiseOpGenerator } from '../generator';
import { OpOutput, OpInput } from '../../commonTypes';
import { OutputDTypeResolver } from '../../dtype';

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

    public static create(generator: ElementWiseOpGenerator): ITrigMathOpSet {

        const opSin = generator.makeUnaryOp({
            opR: '$reY = Math.sin($reX);',
            opC: '$tmp1 = CMath.csin($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opCos = generator.makeUnaryOp({
            opR: '$reY = Math.cos($reX);',
            opC: '$tmp1 = CMath.ccos($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opTan = generator.makeUnaryOp({
            opR: '$reY = Math.tan($reX);',
            opC: '$tmp1 = CMath.ctan($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opCot = generator.makeUnaryOp({
            opR: '$tmp1 = Math.tan($reX); $reY = $tmp1 === 0.0 ? NaN : 1.0 / $tmp1;',
            opC: '$tmp2 = CMath.ccot($reX, $imX); $reY = $tmp2[0]; $imY = $tmp2[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        // TODO: implement complex version of inverse trig functions

        const opAsin = generator.makeUnaryOp({
            opR: '$reY = Math.asin($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opAcos = generator.makeUnaryOp({
            opR: '$reY = Math.acos($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opAtan = generator.makeUnaryOp({
            opR: '$reY = Math.atan($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opAcot = generator.makeUnaryOp({
            opR: '$reY = Math.PI * 0.5 - Math.atan($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opSinh = generator.makeUnaryOp({
            opR: '$reY = 0.5*(Math.exp($reX) - Math.exp(-$reX));',
            opC: '$tmp1 = CMath.csinh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opCosh = generator.makeUnaryOp({
            opR: '$reY = 0.5*(Math.exp($reX) + Math.exp(-$reX));',
            opC: '$tmp1 = CMath.ccosh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opTanh = generator.makeUnaryOp({
            opR: '$tmp1 = Math.exp($reX); $tmp2 = Math.exp(-$reX); $reY = ($tmp1 - $tmp2) / ($tmp1 + $tmp2);',
            opC: '$tmp3 = CMath.ctanh($reX, $imX); $reY = $tmp3[0]; $imY = $tmp3[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opCoth = generator.makeUnaryOp({
            opR: '$tmp1 = Math.exp($reX); $tmp2 = Math.exp(-$reX); $reY = ($tmp1 + $tmp2) / ($tmp1 - $tmp2);',
            opC: '$tmp3 = CMath.ccoth($reX, $imX); $reY = $tmp3[0]; $imY = $tmp3[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
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
            sinh: opSinh,
            cosh: opCosh,
            tanh: opTanh,
            coth: opCoth
        };

    }

}
