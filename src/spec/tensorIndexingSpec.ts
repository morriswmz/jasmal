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
        let BCopy = B.copy(true);

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
        it('should set the first 4 elements to 4 when indices are stored in an array', () => {
            let X = T.zeros([3, 3]);
            X.set([0, 1, 2, 3], 4);
            checkTensor(X, T.fromArray([[4, 4, 4], [4, 0, 0], [0, 0, 0]]));
        });
        it('should set the last 4 elements to 4 when indices are stored in an array', () => {
            let X = T.zeros([3, 3]);
            X.set([-1, -2, -3, -4], 4);
            checkTensor(X, T.fromArray([[0, 0, 0], [0, 0, 4], [4, 4, 4]]));
        });
        it('should set the first 4 elements to 4 when indices are stored in a nested array', () => {
            let X = T.zeros([3, 3]);
            X.set([[0, 1], [2, 3]], 4);
            checkTensor(X, T.fromArray([[4, 4, 4], [4, 0, 0], [0, 0, 0]]));
        });
        it('should set the last 4 elements to 4 when indices are stored in a nested array', () => {
            let X = T.zeros([3, 3]);
            X.set([[-1], [-2], [-3], [-4]], 4);
            checkTensor(X, T.fromArray([[0, 0, 0], [0, 0, 4], [4, 4, 4]]));
        });
        it('should set the first 4 elements correspondingly when indices and values are stored in nested arrays', () => {
            let X = T.zeros([3, 3]);
            X.set([[0, 1], [2, 3]], [[4, 3, 2, 1]]);
            checkTensor(X, T.fromArray([[4, 3, 2], [1, 0, 0], [0, 0, 0]]));
        });
        it('should set the first two and the last two elements to 1+2j via tensor indexing', () => {
            let X = T.zeros([2, 2, 2]);
            let I = T.fromArray([[0, -1], [1, -2]]);
            X.set(I, T.complexNumber(1, 2));
            checkTensor(X, T.fromArray(
                [[[1, 1], [0, 0]], [[0, 0], [1, 1]]],
                [[[2, 2], [0, 0]], [[0, 0], [2, 2]]]
            ));
        });
        it('should set the first two and the last two elements correspondingly via tensor indexing', () => {
            let X = T.zeros([2, 2, 2]);
            let V = T.fromArray([[1], [2], [3], [4]], [[-1], [-2], [-3], [-4]]);
            let I = T.fromArray([[0, -1], [1, -2]]);
            let ICopy = I.copy(true);
            X.set(I, V);
            checkTensor(X, T.fromArray(
                [[[1, 3], [0, 0]], [[0, 0], [4, 2]]],
                [[[-1, -3], [0, 0]], [[0, 0], [-4, -2]]]
            ));
            // should not change I
            checkTensor(I, ICopy);
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
            // should not change A            
            checkTensor(A, ACopy);
        });
        it('should set all the elements that are greater than 2 to -1', () => {
            let X = A.copy();
            X.set(x => x > 2, -1);
            checkTensor(X, T.fromArray([[1, 2, -1], [-1, -1, -1]]));
            // should not change A            
            checkTensor(A, ACopy);
        });
        it('should set all the elements whose imaginary part is less than -2 to 8+9j', () => {
            let X = B.copy();
            X.set((_re, im) => im < -2, T.complexNumber(8, 9));
            let expected = T.fromArray([[1, 2, 8], [8, 8, 8]], [[-1, -2, 9], [9, 9, 9]]);
            checkTensor(X, expected);
            // should not change B            
            checkTensor(B, BCopy);
        });
        it('should use the latest value if there are repeats in the index array', () => {
            let X = T.zeros([2, 2]);
            X.set([0, 1, 1, 0], [1, 2, 3, 4]);
            checkTensor(X, T.fromArray([[4, 3], [0, 0]]));
        });
        it('should throw when indices are not an integer', () => {
            let X = A.copy(true);
            // non-integer in direct indexing
            let case1 = () => { X.set(0, 1.2, -1); };
            // non-integer in an array of indices
            let case2 = () => { X.set([0, -2.2], -1); };
            // non-integer in a tensor of indices
            let case3 = () => { X.set(T.fromArray([[0, 1], [2, 1.5]]), -1); };
            // non-integer in a string
            let case4 = () => { X.set('0.1:', -1); };
            let case5 = () => { X.set('::0.5', -1); };
            let case6 = () => { X.set('-1:0.1:-1', -1); };

            expect(case1).toThrow();
            expect(case2).toThrow();
            expect(case3).toThrow();
            expect(case4).toThrow();
            expect(case5).toThrow();
            expect(case6).toThrow();
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
        it('should remove a singleton dimension if only that dimension is indexed by a number [case 1]', () => {
            let actual = A.get(0, '0:1');
            let expected = T.fromArray([1]);
            checkTensor(actual, expected);
        });
        it('should remove a singleton dimension if only that dimension is indexed by a number [case 2]', () => {
            let actual = A.get('0:1', [2]);
            let expected = T.fromArray([[3]]);
            checkTensor(actual, expected);
        });
        it('should not remove any singleton dimensions if keepDims is set to true [case 1]', () => {
            let actual = A.get(1, 2, true);
            let expected = T.fromArray([[6]]);
            checkTensor(actual, expected);
        });
        it('should not remove any singleton dimensions if keepDims is set to true [case 2]', () => {
            let actual = A.get(1, ':', true);
            let expected = T.fromArray([[4, 5, 6]]);
            checkTensor(actual, expected);
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
            checkTensor(actual, expected);
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
        it('should not change the data type', () => {
            let X = T.ones([3, 3], T.INT32);
            expect((<Tensor>X.get(':', 1)).dtype).toEqual(T.INT32);
            expect((<Tensor>X.get(0, 1, true)).dtype).toEqual(T.INT32);
        });
    });
});