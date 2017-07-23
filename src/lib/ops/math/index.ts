import { IMathOpProvider } from './definition';
import { TensorElementWiseOpCompiler } from '../compiler/compiler';
import { ObjectHelper } from '../../helper/objHelper';
import { BasicMathOpSetFactory } from './basic';
import { TrigMathOpSetFactory } from './trigonometry';
import { PowerMathOpSetFactory } from './pow';
import { LogExpMathOpSetFactory } from './logexp';
import { RoundingMathOpSetFactory } from './rounding';

export class MathOpProviderFactory {

    public static create(): IMathOpProvider {
        
        const compiler = TensorElementWiseOpCompiler.getInstance();

        const notImplemented = () => {
            throw new Error('Not implemented.');
        };

        const basicOps = BasicMathOpSetFactory.create(compiler);
        const trigOps = TrigMathOpSetFactory.create(compiler);
        const powerOps = PowerMathOpSetFactory.create(compiler);
        const logExpOps = LogExpMathOpSetFactory.create(compiler);
        const roundingOps = RoundingMathOpSetFactory.create(compiler);
        
        return ObjectHelper.createExtendChain(basicOps)
            .extend(trigOps)
            .extend(powerOps)
            .extend(logExpOps)
            .extend(roundingOps)
            .end();

    }
}