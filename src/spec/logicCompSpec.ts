import { JasmalEngine } from '..';
import { Tensor } from '../lib/tensor';
import { checkTensor } from './testHelper';
const T = JasmalEngine.createInstance();

describe('eq()', () => {
    it('should test equality between a matrix and a scalar', () => {
        let actual = T.eq([[1, 2], [2, 3]], 2);
        let expected = T.fromArray([[0, 1], [1, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});