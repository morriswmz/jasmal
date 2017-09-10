import { JasmalEngine } from '../index';
import { checkTensor } from './testHelper';
const T = JasmalEngine.createInstance();

let a = T.fromArray([0x3f, 0x83, 0x15], [], T.INT32);
let b = T.fromArray([0x97, 0x16, 0x44], [], T.INT32);
let l = T.fromArray([1, 2, 3], [], T.INT32);

describe('bitwiseAnd()', () => {
    it('should evaluate bitwise AND for two compatible inputs', () => {
        let actual = T.bitwiseAnd(a, b);
        let expected = T.fromArray([0x17, 0x02, 0x04], [], T.INT32);
        checkTensor(actual, expected);
    });
});

describe('bitwiseOr()', () => {
    it('should evaluate bitwise OR for two compatible inputs', () => {
        let actual = T.bitwiseOr(a, b);
        let expected = T.fromArray([0xbf, 0x97, 0x55], [], T.INT32);
        checkTensor(actual, expected);
    });
});

describe('bitwiseXor()', () => {
    it('should evaluate bitwise OR for two compatible inputs', () => {
        let actual = T.bitwiseXor(a, b);
        let expected = T.fromArray([0xa8, 0x95, 0x51], [], T.INT32);
        checkTensor(actual, expected);
    });
});

describe('bitwiseNot()', () => {
    it('should evaluate bitwise NOT for two compatible inputs', () => {
        let actual = T.bitwiseNot(a);
        let expected = T.fromArray([-0x40, -0x84, -0x16], [], T.INT32);
        checkTensor(actual, expected);
    });
});

describe('leftShift()', () => {
    it('should evaluate left shift for two compatible inputs', () => {
        let actual = T.leftShift(a, l);
        let expected = T.fromArray([0x7e, 0x20c, 0xa8], [], T.INT32);
        checkTensor(actual, expected);        
    });
});

describe('rightShiftSP()', () => {
    it('should evaluate sign propagating right shift for two compatible inputs', () => {
        let actual = T.rightShiftSP(0xf0000000 | 0, l);
        let expected = T.fromArray([0xf8000000 | 0, 0xfc000000 | 0, 0xfe000000 | 0]);
        checkTensor(actual, expected);        
    });
});

describe('rightShiftZF()', () => {
    it('should evaluate zero filling right shift for two compatible inputs', () => {
        let actual = T.rightShiftZF(0xf0000000 | 0, l);
        let expected = T.fromArray([0x78000000 | 0, 0x3c000000 | 0, 0x1e000000 | 0]);
        checkTensor(actual, expected);        
    });
});
