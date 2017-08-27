import { IMathOpProvider } from './definition';
import { ElementWiseOpGenerator } from '../generator';
import { ObjectHelper } from '../../helper/objHelper';
import { BasicMathOpSetFactory } from './basic';
import { TrigMathOpSetFactory } from './trigonometry';
import { PowerMathOpSetFactory } from './pow';
import { LogExpMathOpSetFactory } from './logexp';
import { RoundingMathOpSetFactory } from './rounding';
import { SpecialFunctionOpSetFactory } from './special';

export class MathOpProviderFactory {

    public static create(generator: ElementWiseOpGenerator): IMathOpProvider {
        
        const basicOps = BasicMathOpSetFactory.create(generator);
        const trigOps = TrigMathOpSetFactory.create(generator);
        const powerOps = PowerMathOpSetFactory.create(generator);
        const logExpOps = LogExpMathOpSetFactory.create(generator);
        const roundingOps = RoundingMathOpSetFactory.create(generator);
        const specialOps = SpecialFunctionOpSetFactory.create(generator);
        
        return ObjectHelper.createExtendChain(basicOps)
            .extend(trigOps)
            .extend(powerOps)
            .extend(logExpOps)
            .extend(roundingOps)
            .extend(specialOps)
            .end();

    }
}
