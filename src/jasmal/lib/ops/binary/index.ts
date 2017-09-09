import { IBinaryOpProvider } from './definition';
import { ElementWiseOpGenerator } from '../generator/index';
import { OutputDTypeResolver } from '../../dtype';

export class BinaryOpProviderFactory {

    public static create(generator: ElementWiseOpGenerator): IBinaryOpProvider {

        const opBitwiseAnd = generator.makeBinaryOp({
            opRR: '$reZ = $reX & $reY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt,
        });

        const opBitwiseOr = generator.makeBinaryOp({
            opRR: '$reZ = $reX | $reY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt,
        });

        const opBitwiseXor = generator.makeBinaryOp({
            opRR: '$reZ = $reX ^ $reY;'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt,
        });

        const opBitwiseNot = generator.makeUnaryOp({
            opR: '$reY = ~($reX);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.uNoChangeExceptLogicToInt,
        });

        const opLeftShift = generator.makeBinaryOp({
            opRR: '$reZ = ($reX) << ($reY);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });

        const opRightShiftSP= generator.makeBinaryOp({
            opRR: '$reZ = ($reX) >> ($reY);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });

        const opRightShiftZF = generator.makeBinaryOp({
            opRR: '$reZ = ($reX) >>> ($reY);'
        }, {
            outputDTypeResolver: OutputDTypeResolver.bWiderWithLogicToInt
        });

        
        return {
            bitwiseAnd: opBitwiseAnd,
            bitwiseOr: opBitwiseOr,
            bitwiseXor: opBitwiseXor,
            bitwiseNot: opBitwiseNot,
            leftShift: opLeftShift,
            rightShiftSP: opRightShiftSP,
            rightShiftZF: opRightShiftZF,
        };

    }

}

