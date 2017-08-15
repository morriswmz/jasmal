import { IBasicMathOpSet } from './basic';
import { ITrigMathOpSet } from './trigonometry';
import { IPowerMathOpSet } from './pow';
import { ILogExpMathOpSet } from './logexp';
import { IRoundingMathOpSet } from './rounding';

export interface IMathOpProvider extends IBasicMathOpSet, ITrigMathOpSet,
    IPowerMathOpSet, ILogExpMathOpSet, IRoundingMathOpSet {}