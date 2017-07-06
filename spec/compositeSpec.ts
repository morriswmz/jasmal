import {JasmalEngine} from '..';
import { Tensor } from "../lib/tensor";

let T = JasmalEngine.createInstance();
T.seed(192);

describe('Composite problems > ', () => {
    it('broadcasting and tile', () => {
        let x = T.rand([3, 1]), y = T.rand([1, 3]);
        let z1 = <Tensor>T.add(x, y);
        let z2 = <Tensor>T.add(T.tile(x, [1, 3]), T.tile(y, [3, 1]));
        expect(z1.shape).toEqual(z2.shape);
        expect(z1.realData).toEqual(z2.realData);
    });
});