import { JasmalEngine } from '../index';
import { checkTensor } from './testHelper';
const T = JasmalEngine.createInstance();

describe('unique()', () => {
    it('should find unique real numbers', () => {
        let x = T.fromArray([3, 4, 3, 2, 3, -1, NaN, -1, Infinity, NaN]);
        let expectedY = T.fromArray([-1, 2, 3, 4, Infinity, NaN, NaN]);
        let expectedIy = [5, 3, 0, 1, 8, 6, 9];
        let expectedIx = [[5, 7], [3], [0, 2, 4], [1], [8], [6], [9]];
        let [actualY, actualIy, actualIx] = T.unique(x, true);
        checkTensor(actualY, expectedY);
        expect(actualIy).toEqual(expectedIy);
        expect(actualIx).toEqual(expectedIx);
    });
    it('should find unique complex numbers', () => {
        let x = T.fromArray(
            [3, 3, 3, 3, -1, -1, Infinity, NaN,   0],
            [2, 2, 5, 5, -2, -2,        0,   2, NaN]);
        let expectedY = T.fromArray(
            [-1,   0, 3, 3, Infinity, NaN],
            [-2, NaN, 2, 5,        0,   2]);
        let expectedIy = [4, 8, 0, 2, 6, 7];
        let expectedIx = [[4, 5], [8], [0, 1], [2, 3], [6], [7]];
        let [actualY, actualIy, actualIx] = T.unique(x, true);
        checkTensor(actualY, expectedY);
        expect(actualIy).toEqual(expectedIy);
        expect(actualIx).toEqual(expectedIx);
    });
});
