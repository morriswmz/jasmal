import { ElementWiseOpGenerator } from '../generator';
import { RealOpInput, RealOpOutput } from '../../commonTypes';
import { OutputDTypeResolver } from '../../dtype';
import { SpecialFunction } from '../../math/special';

/**
 * A collection of special functions.
 */
export interface ISpecialFunctionOpSet {

    /**
     * Logarithm of gamma function.
     */
    gammaln(x: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Gamma function.
     */
    gamma(x: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Factorial function.
     */
    factorial(x: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Error function.
     */
    erf(x: RealOpInput, inPlace?: boolean): RealOpOutput;

    /**
     * Complementary error function.
     */
    erfc(x: RealOpInput, inPlace?: boolean): RealOpOutput;
    
    /**
     * Scaled complementary error function: exp(x^2) * erfc(x).
     */
    erfcx(x: RealOpInput, inPlace?: boolean): RealOpOutput;

}

export class SpecialFunctionOpSetFactory {

    public static create(generator: ElementWiseOpGenerator): ISpecialFunctionOpSet {

        const opGammaLn = generator.makeRealOutputUnaryOp({
            opR: '$reY = SpecialFunction.gammaln($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'SpecialFunction': SpecialFunction }
        });

        const opGamma = generator.makeRealOutputUnaryOp({
            opR: '$reY = SpecialFunction.gamma($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'SpecialFunction': SpecialFunction }
        });

        const opFactorial = generator.makeRealOutputUnaryOp({
            opR: '$reY = SpecialFunction.factorial($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat,
            extraDependencies: { 'SpecialFunction': SpecialFunction }
        });

        const opErf = generator.makeRealOutputUnaryOp({
            opR: '$reY = SpecialFunction.erf($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'SpecialFunction': SpecialFunction }
        });

        const opErfc = generator.makeRealOutputUnaryOp({
            opR: '$reY = SpecialFunction.erfc($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'SpecialFunction': SpecialFunction }
        });

        const opErfcx = generator.makeRealOutputUnaryOp({
            opR: '$reY = SpecialFunction.erfcx($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'SpecialFunction': SpecialFunction }
        });

        return {
            gammaln: opGammaLn,
            gamma: opGamma,
            factorial: opFactorial,
            erf: opErf,
            erfc: opErfc,
            erfcx: opErfcx
        };

    }

}
