import {JasmalEngine} from '..';
import { Tensor } from '../lib/tensor';
import { checkTensor } from './testHelper';

let T = JasmalEngine.createInstance();
T.seed(192);

describe('Composite problems', () => {
    it('broadcasting and tile', () => {
        let x = T.rand([3, 1]), y = T.rand([1, 3]);
        let z1 = <Tensor>T.add(x, y);
        let z2 = <Tensor>T.add(T.tile(x, [1, 3]), T.tile(y, [3, 1]));
        expect(z1.shape).toEqual(z2.shape);
        expect(z1.realData).toEqual(z2.realData);
    });
    it('changing image dimension ordering should not change the content', () => {
        let images = T.rand([2, 64, 48, 3]); // two 64x48 rgb random noise
        // [n, w, h, c] -> [n, c, w, h]
        let reorderedImages = T.permuteAxis(images, [0, 3, 1, 2]);
        let expected = <Tensor>images.get(0, ':', ':', 0);
        let actual = reorderedImages.get(0, 0, ':', ':');
        checkTensor(actual, expected);
    });
});