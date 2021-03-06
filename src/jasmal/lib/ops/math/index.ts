import { IMathOpProvider } from './definition';
import { ElementWiseOpGenerator } from '../generator';
import { ObjectHelper } from '../../helper/objHelper';
import { BasicMathOpSetFactory } from './basic';
import { TrigMathOpSetFactory } from './trigonometry';
import { PowerMathOpSetFactory } from './pow';
import { LogExpMathOpSetFactory } from './logexp';
import { RoundingMathOpSetFactory } from './rounding';
import { SpecialFunctionOpSetFactory } from './special';
import { IJasmalModuleFactory, JasmalOptions } from '../../jasmal';

export class MathOpProviderFactory implements IJasmalModuleFactory<IMathOpProvider> {
    
    constructor(private _generator: ElementWiseOpGenerator) {
    }

    public create(_options: JasmalOptions): IMathOpProvider {
        
        const generator = this._generator;

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
