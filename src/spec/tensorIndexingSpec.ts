import { JasmalEngine } from '../';
import { ComplexNumber } from '../lib/complexNumber';
import { Tensor } from '../lib/tensor';
import { checkTensor } from './testHelper';
const T = JasmalEngine.createInstance();

describe('Advanced indexing', () => {
    describe('set()', () => {
        let A = T.fromArray([[1, 2, 3], [4, 5, 6]]);
        let ACopy = A.copy(true);
        let B = T.fromArray([[1, 2, 3], [4, 5, 6]], [[-1, -2, -3], [-4, -5, -6]]);

        it('should set all elements to 1', () => {
            let X = T.zeros([3, 4]);
            X.set(':', 1);
            checkTensor(X, T.ones(X.shape));
        });
        it('should set masked elements to 7', () => {
            let mask = T.fromArray([[1, 1, 0], [0, 1, 1]], [], T.LOGIC);
            let X = A.copy();
            X.set(mask, 7);
            checkTensor(X, T.fromArray([[7, 7, 3], [4, 7, 7]]));
            checkTensor(A, ACopy); // should not change A
        });
        it('should set all elements at even indices to 1', () => {
            let X = T.zeros([3, 3]);
            X.set('::2', 1);
            checkTensor(X, T.fromArray([[1, 0, 1], [0, 1, 0], [1, 0, 1]]));
        });
        it('should set four corners to [1,2,3,4]', () => {
            let X = T.zeros([3, 3]);
            X.set([0, -1], [0, -1], [[1, 2], [3, 4]]);
            checkTensor(X, T.fromArray([[1, 0, 2], [0, 0, 0], [3, 0, 4]]));
        });
        it('should set all elements in the first row to 10', () => {
            let X = A.copy();
            X.set(0, ':', 10);
            checkTensor(X, T.fromArray([[10, 10, 10], [4, 5, 6]]));
        });
        it('should set all elements that are greater than 2 to -1', () => {
            let X = A.copy();
            X.set(x => x > 2, -1);
            checkTensor(X, T.fromArray([[1, 2, -1], [-1, -1, -1]]));
            // should not change A            
            checkTensor(A, ACopy);
        });

    });
    describe('get()', () => {
        let A = T.fromArray([[1, 2, 3], [4, 5, 6]]);
        let B = T.fromArray(
            [[-1, 0, 0, -2],
             [0, 20, 30, 0],
             [0 ,40, 50, 0],
             [-3, 0, 0, -4]]);
        let C = T.fromArray([[1, 2, 3], [4, 5, 6]], [[-1, -2, -3], [-4, -5, -6]]);

        it('should get a single element using flat indexing', () => {
            expect(A.get(1)).toBe(2);
        });
        it('should get a single element using (i,j) indexing', () => {
            expect(A.get(0, 1)).toBe(2);
        });
        it('should get a single element using (i,j) indexing with keepDims = true', () => {
            let actual = A.get(0, 1, true);
            let expected = T.fromArray([[2]]);
            checkTensor(actual, expected);
        });
        it('should get a single complex element using flat indexing with a negative index', () => {
            expect((<ComplexNumber>C.get(-1)).equals(new ComplexNumber(6, -6))).toBeTruthy();
        });
        it('should get a single complex element using (i,j) indexing with negative indices', () => {
            expect((<ComplexNumber>C.get(-1, 0)).equals(new ComplexNumber(4, -4))).toBeTruthy();
        });
        it('should return a tensor with all the elements reversed', () => {
            let actual = <Tensor>A.get('::-1');
            let expected = T.fromArray([6, 5, 4, 3, 2, 1]);
            checkTensor(actual, expected);
        });
        it('should return a tensor with all the rows reversed', () => {
            let actual = <Tensor>C.get('::-1',':');
            let expected = T.fromArray([[4, 5, 6], [1, 2, 3]], [[-4, -5, -6], [-1, -2, -3]]);
            checkTensor(actual, expected);
        });
        it('should return a sub matrix', () => {
            let actual = B.get([1, 2], '1:3');
            let expected = T.fromArray([[20, 30], [40, 50]]);
            checkTensor(actual, expected);
        });
        it('should return a sub tensor', () => {
            let M = T.fromArray(
                [[[1, 2, 3],
                  [4, 5, 6]],
                 [[7, 8, 9],
                  [10, 11, 12]]]
            );
            let actual = M.get(':', '::-1', [0, -1]);
            let expected = T.fromArray(
                [[[4, 6],
                  [1, 3]],
                 [[10, 12],
                  [7, 9]]],
            );
        });
        it('should return four corners using masked indexing', () => {
            let actual = A.get(T.fromArray([1, 1], [], T.LOGIC), T.fromArray([1, 0, 1], [], T.LOGIC));
            let expected = T.fromArray([[1, 3], [4, 6]]);
            checkTensor(actual, expected);
        });
        it('should return all negative elements', () => {
            let actual = B.get(x => x < 0);
            let expected = T.fromArray([-1, -2, -3, -4]);
            checkTensor(actual, expected);
        });
        it('should sample columns', () => {
            let actual = A.get(':', [0, -1, 0, -1, 1]);
            let expected = T.fromArray(
                [[1, 3, 1, 3, 2],
                 [4, 6, 4, 6, 5]]);
            checkTensor(actual, expected);
        });
        it('should form a new matrix when the index input is a matrix', () => {
            let actual = A.get([[0, 1], [1, -1]]);
            let expected = T.fromArray([[1, 2], [2, 6]]);
            checkTensor(actual, expected);
        });
    });
});