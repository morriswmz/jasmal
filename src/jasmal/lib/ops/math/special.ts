import { ElementWiseOpGenerator } from '../generator';
import { OpInput, OpOutput } from '../../commonTypes';
import { OutputDTypeResolver } from '../../dtype';
import { SpecialFunction } from '../../math/special';

/**
 * A collection of special functions.
 */
export interface ISpecialFunctionOpSet {

    gammaln(x: OpInput, inPlace?: boolean): OpOutput;

    gamma(x: OpInput, inPlace?: boolean): OpOutput;

    factorial(x: OpInput, inPlace?: boolean): OpOutput;

    erf(x: OpInput, inPlace?: boolean): OpOutput;

    erfc(x: OpInput, inPlace?: boolean): OpOutput;
    
    erfcx(x: OpInput, inPlace?: boolean): OpOutput;

}

export class SpecialFunctionOpSetFactory {

    public static create(generator: ElementWiseOpGenerator): ISpecialFunctionOpSet {

        const opGammaLn = generator.makeUnaryOp({
            opR: '$reY = SpecialFunction.gammaln($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'SpecialFunction': SpecialFunction }
        });

        const opGamma = generator.makeUnaryOp({
            opR: '$reY = SpecialFunction.gamma($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'SpecialFunction': SpecialFunction }
        });

        const opFactorial = generator.makeUnaryOp({
            opR: '$reY = SpecialFunction.factorial($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uOnlyLogicToFloat,
            extraDependencies: { 'SpecialFunction': SpecialFunction }
        });

        const opErf = generator.makeUnaryOp({
            opR: '$reY = SpecialFunction.erf($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'SpecialFunction': SpecialFunction }
        });

        const opErfc = generator.makeUnaryOp({
            opR: '$reY = SpecialFunction.erfc($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uToFloat,
            extraDependencies: { 'SpecialFunction': SpecialFunction }
        });

        const opErfcx = generator.makeUnaryOp({
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