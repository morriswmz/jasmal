import { ElementWiseOpGenerator } from '../generator';
import { OpOutput, OpInput } from '../../commonTypes';
import { OutputDTypeResolver } from '../../dtype';
import { HyperbolicTrigonometry } from '../../math/hyptrig';
import { Tensor } from '../../tensor';
import { ComplexNumber } from "../../complexNumber";
import { CMath } from '../../math/cmath';
import { M_PI_2 } from '../../constant';

export interface ITrigMathOpSet {

    /**
     * Compute the sine for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    sin(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the cosine for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    cos(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the tangent for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    tan(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the cotangent for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    cot(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the hyperbolic sine for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    sinh(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the hyperbolic cosine for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    cosh(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the hyperbolic tangent for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    tanh(x: OpInput, inPlace?: boolean): OpOutput;
    
    /**
     * Compute the hyperbolic cotangent for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    coth(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the inverse sine for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    asin(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the inverse cosine for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    acos(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the inverse tangent for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    atan(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the inverse cotangent for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    acot(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the inverse hyperbolic sine for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    asinh(x: OpInput, inPlace?: boolean): OpOutput;

    /**
     * Compute the inverse hyperbolic cosine for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    acosh(x: OpInput, inPlace?: boolean): OpOutput;
    
    /**
     * Compute the inverse hyperbolic tangent for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    atanh(x: OpInput, inPlace?: boolean): OpOutput;
    
    /**
     * Compute the inverse hyperbolic cotangent for each element in the input.
     * @param x Input.
     * @param inPlace (Optional) Whether this operation should be performed in
     *                place. Default value is false.
     */
    acoth(x: OpInput, inPlace?: boolean): OpOutput;
    
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

    public static anyAbsLessThanOne(x: ArrayLike<number>): boolean {
        for (let i = 0;i < x.length;i++) {
            if (x[i] < 1 && x[i] > -1) {
                return true;
            }
        }
        return false;
    }

    public static anyLessThanOne(x: ArrayLike<number>): boolean {
        for (let i = 0;i < x.length;i++) {
            if (x[i] < 1) {
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
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
        });

        const opCos = generator.makeUnaryOp({
            opR: '$reY = Math.cos($reX);',
            opC: '$tmp1 = CMath.ccos($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
        });

        const opTan = generator.makeUnaryOp({
            opR: '$reY = Math.tan($reX);',
            opC: '$tmp1 = CMath.ctan($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
        });

        const opCot = generator.makeUnaryOp({
            opR: '$tmp1 = Math.tan($reX); $reY = $tmp1 === 0.0 ? NaN : 1.0 / $tmp1;',
            opC: '$tmp2 = CMath.ccot($reX, $imX); $reY = $tmp2[0]; $imY = $tmp2[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
        });

        const opAsinR = generator.makeUnaryOp({
            opR: '$reY = Math.asin($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
        });
        
        const opAsinC = generator.makeUnaryOp({
            opR: '$tmp1 = CMath.casin($reX, 0); $reY = $tmp1[0]; $imY = $tmp1[1];',
            opC: '$tmp1 = CMath.casin($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
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
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
        });

        const opAcosC = generator.makeUnaryOp({
            opR: '$tmp1 = CMath.cacos($reX, 0); $reY = $tmp1[0]; $imY = $tmp1[1];',
            opC: '$tmp1 = CMath.cacos($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
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
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
        });

        const opAcot = generator.makeUnaryOp({
            opR: '$tmp1 = Math.atan($reX); $reY = ($tmp1 >= 0 ? M_PI_2 : -M_PI_2) - $tmp1; ',
            opC: '$tmp1 = CMath.cacot($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64,
            extraDependencies: { 'M_PI_2': M_PI_2 }
        });

        const opSinh = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.sinh($reX);',
            opC: '$tmp1 = CMath.csinh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }
        });

        const opCosh = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.cosh($reX);',
            opC: '$tmp1 = CMath.ccosh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }            
        });

        const opTanh = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.tanh($reX);',
            opC: '$tmp3 = CMath.ctanh($reX, $imX); $reY = $tmp3[0]; $imY = $tmp3[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }
        });

        const opCoth = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.coth($reX);',
            opC: '$tmp3 = CMath.ccoth($reX, $imX); $reY = $tmp3[0]; $imY = $tmp3[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }
        });

        const opAsinh = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.asinh($reX);',
            opC: '$tmp1 = CMath.casinh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }
        });

        const opAcoshR = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.acosh($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }
        });

        const opAcoshC = generator.makeUnaryOp({
            opR: '$tmp1 = CMath.cacosh($reX, 0); $reY = $tmp1[0]; $imY = $tmp1[1];',
            opC: '$tmp1 = CMath.cacosh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
        });

        const opAcosh = (x: OpInput, inPlace: boolean = false): OpOutput => {
            let infoX = Tensor.analyzeOpInput(x);
            if (infoX.isInputScalar) {
                if (inPlace) {
                    throw new Error('Cannot perform in-place operation for a scalar input.');
                }
                let [re, im] = CMath.cacosh(infoX.re, infoX.im);
                return im === 0 ? re : new ComplexNumber(re, im);
            } else {
                if (infoX.isComplex || InputRangeChecker.anyLessThanOne(infoX.reArr)) {
                    return opAcoshC(infoX, inPlace);
                } else {
                    return opAcoshR(infoX, inPlace);
                }
            }
        };

        const opAtanhR = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.atanh($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }
        });

        const opAtanhC = generator.makeUnaryOp({
            opR: '$tmp1 = CMath.catanh($reX, 0); $reY = $tmp1[0]; $imY = $tmp1[1];',
            opC: '$tmp1 = CMath.catanh($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
        });

        const opAtanh = (x: OpInput, inPlace: boolean = false): OpOutput => {
            let infoX = Tensor.analyzeOpInput(x);
            if (infoX.isInputScalar) {
                if (inPlace) {
                    throw new Error('Cannot perform in-place operation for a scalar input.');
                }
                let [re, im] = CMath.catanh(infoX.re, infoX.im);
                return im === 0 ? re : new ComplexNumber(re, im);
            } else {
                if (infoX.isComplex || InputRangeChecker.anyAbsGreaterThanOne(infoX.reArr)) {
                    return opAtanhC(infoX, inPlace);
                } else {
                    return opAtanhR(infoX, inPlace);
                }
            }
        };

        const opAcothR = generator.makeUnaryOp({
            opR: '$reY = HyperbolicTrigonometry.acoth($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64,
            extraDependencies: { 'HyperbolicTrigonometry': HyperbolicTrigonometry }
        });

        const opAcothC = generator.makeUnaryOp({
            opR: '$tmp1 = CMath.cacoth($reX, 0); $reY = $tmp1[0]; $imY = $tmp1[1];',
            opC: '$tmp1 = CMath.cacoth($reX, $imX); $reY = $tmp1[0]; $imY = $tmp1[1];'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat64
        });

        const opAcoth = (x: OpInput, inPlace: boolean = false): OpOutput => {
            let infoX = Tensor.analyzeOpInput(x);
            if (infoX.isInputScalar) {
                if (inPlace) {
                    throw new Error('Cannot perform in-place operation for a scalar input.');
                }
                let [re, im] = CMath.cacoth(infoX.re, infoX.im);
                return im === 0 ? re : new ComplexNumber(re, im);
            } else {
                if (infoX.isComplex || InputRangeChecker.anyAbsLessThanOne(infoX.reArr)) {
                    return opAcothC(infoX, inPlace);
                } else {
                    return opAcothR(infoX, inPlace);
                }
            }
        };

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
            coth: opCoth,
            asinh: opAsinh,
            acosh: opAcosh,
            atanh: opAtanh,
            acoth: opAcoth
        };

    }

}
