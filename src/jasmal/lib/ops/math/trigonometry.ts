import { ElementWiseOpGenerator } from '../generator';
import { OpOutput, OpInput } from '../../commonTypes';
import { OutputDTypeResolver } from '../../dtype';
import { HyperbolicTrigonometry } from '../../math/hyptrig';
import { Tensor } from '../../tensor';
import { ComplexNumber } from "../../complexNumber";
import { CMath } from '../../math/cmath';
import { M_PI_2 } from '../../constant';

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

class InputRangeChecker {

    public static anyAbsGreaterThanOne(x: ArrayLike<number>): boolean {
        for (let i = 0;i < x.length;i++) {
            // NaN > 1 is false, NaN < -1 is false
            if (x[i] > 1 || x[i] < -1) {
                return true;
            }
        }
        return false;
    }
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

        const opAsinR = generator.makeUnaryOp({
            opR: '$reY = Math.asin($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });
        
        const opAsinC = generator.makeUnaryOp({
            opR: '$tmp1 = CMath.casin($reX, 0); $reY = $tmp1[0]; $imY = $tmp1[1];',
            opC: '$tmp1 = CMath.casin($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opAsin = (x: OpInput, inPlace: boolean = false): OpOutput => {
            let infoX = Tensor.analyzeOpInput(x);
            if (infoX.isInputScalar) {
                if (inPlace) {
                    throw new Error('Cannot perform in-place operation for a scalar input.');
                }
                let [re, im] = CMath.casin(infoX.re, infoX.im);
                return im === 0 ? re : new ComplexNumber(re, im);
            } else {
                if (infoX.isComplex || InputRangeChecker.anyAbsGreaterThanOne(infoX.reArr)) {
                    return opAsinC(infoX, inPlace);
                } else {
                    return opAsinR(infoX, inPlace);
                }
            }
        };

        const opAcosR = generator.makeUnaryOp({
            opR: '$reY = Math.acos($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opAcosC = generator.makeUnaryOp({
            opR: '$tmp1 = CMath.cacos($reX, 0); $reY = $tmp1[0]; $imY = $tmp1[1];',
            opC: '$tmp1 = CMath.cacos($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opAcos = (x: OpInput, inPlace: boolean = false): OpOutput => {
            let infoX = Tensor.analyzeOpInput(x);
            if (infoX.isInputScalar) {
                if (inPlace) {
                    throw new Error('Cannot perform in-place operation for a scalar input.');
                }
                let [re, im] = CMath.cacos(infoX.re, infoX.im);
                return im === 0 ? re : new ComplexNumber(re, im);
            } else {
                if (infoX.isComplex || InputRangeChecker.anyAbsGreaterThanOne(infoX.reArr)) {
                    return opAcosC(infoX, inPlace);
                } else {
                    return opAcosR(infoX, inPlace);
                }
            }
        };

        const opAtan = generator.makeUnaryOp({
            opR: '$reY = Math.atan($reX);',
            opC: '$tmp1 = CMath.catan($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat
        });

        const opAcot = generator.makeUnaryOp({
            opR: '$tmp1 = Math.atan($reX); $reY = ($tmp1 >= 0 ? M_PI_2 : -M_PI_2) - $tmp1; ',
            opC: '$tmp1 = CMath.cacot($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: {
                'M_PI_2': M_PI_2
            }
        });

        const opSinh = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.sinh($reX);',
            opC: '$tmp1 = CMath.csinh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }
        });

        const opCosh = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.cosh($reX);',
            opC: '$tmp1 = CMath.ccosh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }            
        });

        const opTanh = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.tanh($reX);',
            opC: '$tmp3 = CMath.ctanh($reX, $imX); $reY = $tmp3[0]; $imY = $tmp3[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }
        });

        const opCoth = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.coth($reX);',
            opC: '$tmp3 = CMath.ccoth($reX, $imX); $reY = $tmp3[0]; $imY = $tmp3[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }
        });

        // TODO: implement inverse hyptrig functions

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
