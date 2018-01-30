import { JasmalEngine } from '../index';
import { Tensor } from '../lib/core/tensor';
import { checkTensor } from './testHelper';
import { EPSILON } from '../lib/constant';

let T = JasmalEngine.createInstance();
T.seed(192);

describe('Composite problems', () => {
    it('matrix multiplication is associative', () => {
        let A = T.randn([20, 18]);
        let B = T.complex(T.randn([20, 18]), T.randn([20, 18]));
        let C = T.randn([18, 30]);
        let X1 = T.add(T.matmul(A, C), T.matmul(B, C));
        let X2 = T.matmul(T.add(A, B), C);
        checkTensor(X1, X2, 1e-14);
    });
    it('broadcasting and tile', () => {
        let x = T.rand([3, 1]), y = T.rand([1, 3]);
        let z1 = T.add(x, y);
        let z2 = T.add(T.tile(x, [1, 3]), T.tile(y, [3, 1]));
        checkTensor(T.sub(z1, z2), T.zeros([x.shape[0], y.shape[1]]));
    });
    it('outer product can also be performed via broadcasting', () => {
        let v = T.rand([10, 1]);
        let W1 = T.matmul(v, v, T.MM_TRANSPOSED);
        let W2 = T.mul(v, T.transpose(v));
        checkTensor(T.sub(W1, W2), T.zeros([v.size, v.size]));
    });
    it('changing image dimension ordering should not change the content', () => {
        let images = T.rand([2, 64, 48, 3]); // two 64x48 rgb random noise
        // [n, w, h, c] -> [n, c, w, h]
        let reorderedImages = T.permuteAxis(images, [0, 3, 1, 2]);
        let expected = <Tensor>images.get(0, ':', ':', 0);
        let actual = reorderedImages.get(0, 0, ':', ':');
        checkTensor(actual, expected);
    });
    it('sin^2(x) + cos^(x) = 1', () => {
        let x = T.rand([10, 10]);
        let actual = T.add(T.square(T.sin(x)), T.square(T.cos(x)));
        let expected = T.ones(x.shape);
        checkTensor(actual, expected, EPSILON);
    });
    it('z * z = z^2', () => {
        let x = T.complex(T.randn([100]), T.randn([100]));
        let xCopy = x.copy(true);
        let y1 = x.copy();
        T.mul(y1, y1, true);
        let y2 = x.copy();
        T.square(y2, true);
        checkTensor(x, xCopy);
        checkTensor(y1, y2);
    });
});
