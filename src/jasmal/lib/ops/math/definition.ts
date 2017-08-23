import { IBasicMathOpSet } from './basic';
import { ITrigMathOpSet } from './trigonometry';
import { IPowerMathOpSet } from './pow';
import { ILogExpMathOpSet } from './logexp';
import { IRoundingMathOpSet } from './rounding';
import { ISpecialFunctionOpSet } from './special';

export interface IMathOpProvider extends IBasicMathOpSet, ITrigMathOpSet,
    IPowerMathOpSet, ILogExpMathOpSet, IRoundingMathOpSet, ISpecialFunctionOpSet {}