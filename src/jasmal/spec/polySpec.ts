import { JasmalEngine } from '..';
import { checkTensor } from './testHelper';
const T = JasmalEngine.createInstance();

describe('polyval()', () => {
    let p1 = [3, 2, -1];
    it('should evaluate a polynomial with real coefficients for a scalar input', () => {
        expect(T.polyval(p1, 2)).toEqual(15);    
    });
    it('should evaluate a polynomial with real coefficients for a matrix input', () => {
        let actual = T.polyval(p1, [[-1, 0], [3, -5]]);
        let expected = T.fromArray([[0, -1], [32, 64]]);
        checkTensor(actual, expected);
    });
});

describe('polyvalm()', () => {
    let p1 = [1, 2, -1, 3, 5];
    let A1 = T.fromArray([[-1, 2, 0], [-2, 3, 1], [-3, 5, 2]]);
    it('should behave like polyval() if the input x is a scalar', () => {
        expect(T.polyvalm(p1, 2)).toEqual(39);
    });
    it('should evaluate a matrix polynomial with real coefficients for a real matrix', () => {
        let actual = T.polyvalm(p1, A1);
        let expected = T.fromArray(
            [ [-62, 100,  46],
             [-169, 253, 119],
             [-311, 457, 226]]);
        checkTensor(actual, expected);
    });
});