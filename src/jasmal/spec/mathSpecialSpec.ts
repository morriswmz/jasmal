import { JasmalEngine } from '../index';
import { checkTensor } from './testHelper';
const T = JasmalEngine.createInstance();

describe('factorial()', () => {
    it('should calculate the factorial of the input', () => {
        let s = 1;
        let x: number[] = [0];
        let y: number[] = [1];
        for (let i = 1;i <= 170;i++) {
            s *= i;
            x.push(i);
            y.push(s);
        }
        x = x.concat([171, 200, 5000, 19923]);
        y = y.concat([Infinity, Infinity, Infinity, Infinity]);
        let actual = T.factorial(x);
        let expected = T.fromArray(y);
        checkTensor(actual, expected);
    });
    it('should not change the data type', () => {
        let x = T.fromArray([1, 3, 2], [], T.INT32);
        let actual = T.factorial(x);
        let expected = T.fromArray([1, 6, 2], [], T.INT32);
        checkTensor(actual, expected);
    });
    it('should throw for invalid inputs', () => {
        expect(() => T.factorial(-1)).toThrow();
        expect(() => T.factorial(NaN)).toThrow();
        expect(() => T.factorial(2.3)).toThrow();
        expect(() => T.factorial(T.fromArray([1, 2], [2, 3]))).toThrow();
    });
});

describe('gammaln()', () => {
    it('should compute log(gamma(x)) for the input x', () => {
        expect(T.gammaln(0)).toBe(Infinity);
        expect(T.gammaln(Infinity)).toBe(Infinity);
        let actual = T.gammaln([
            1e-30,
            0.1,
            0.4,
            0.55,
            0.8,
            1.4,
            2.5,
            3.9,
            7.3,
            11.2,
            12.0,
            14.4,
            28.8,
            112
        ]);
        let expected = T.fromArray([
            69.0775527898213681,
            2.2527126517342055,
            0.79667781770178370,
            0.48003085611112595,
            0.15205967839983756,
           -0.11961291417237130,
            0.28468287047291918,
            1.6675803472417401,
            7.1478925230222501,
            15.576654464525772,
            17.502307845873887,
            23.599196712735910,
            67.220455391594982,
            415.03230672824964
        ]);
        checkTensor(actual, expected);
    });
});

describe('gamma()', () => {
    it('should compute gamma(x) for the input x', () => {
        let actual = T.gamma([
            -20.5,
            -10.2,
            -1.5,
            -0.3,
            -1e-18,
            0,
            1e-18,
            0.4,
            1.2,
            5,
            9.3,
            14.4,
            36,
            182
        ]);
        let expected = T.fromArray([
            -2.8346565743913187e-19,
            -9.1849354167820546e-07,
             2.3632718012073548,
            -4.326851108825192,
            -1e+18,
             Infinity,
             9.9999999999999987e+17,
             2.2181595437576882,
             0.91816874239976065,
             24,
             7.7035557963696396e+04,
             1.7741931971821987e+10,
             1.0333147966386149e+40,
             Infinity
        ]);
        checkTensor(actual, expected, 15, false);
    });
});

describe('erf()/erfc()/erfcx()', () => {
    let x = [-2e300, -50, -11.2, -4.4, -3.8, -0.23, 0, 0.14, 0.5, 1.2, 5.7, 22.5, 56, 2e150];
    it('should evaluate the error function', () => {
        let actual = T.erf(x);
        let expected = T.fromArray([
            -1,
            -1,
            -1,
            -9.9999999951082896e-1,
            -9.9999992299607254e-1,
            -2.5502259959227319e-1,
             0,
             1.5694703306285582e-1,
             5.2049987781304652e-1,
             9.1031397822963533e-1,
             9.9999999999999922e-1,
             1,
             1,
             1
        ]);
        checkTensor(actual, expected, 14, false);
    });
    it('should evaluate the complementary error function', () => {
        let actual = T.erfc(x);
        let expected = T.fromArray([
            2,
            2,
            2,
            1.9999999995108291,
            1.9999999229960725,
            1.2550225995922732,
            1,
            8.4305296693714415e-01,
            4.7950012218695348e-01,
            8.9686021770364638e-02,
            7.5662116218624866e-16,
            3.4453488604646018e-222,
            0,
            0
        ]);
        checkTensor(actual, expected, 14, false);
    });
    it('should evaluate the scaled complementary error function', () => {
        let actual = T.erfcx(x);
        let expected = T.fromArray([
            Infinity,
            Infinity,
            6.0107657848928393e+54,
            5.1164786377275527e+8,
            3.7345845622817636e+6,
            1.3232007076178662,
            1,
            8.5973980187370158e-1,
            6.1569034419292590e-1,
            3.7853741692923976e-1,
            9.7522808795439661e-2,
            2.5050400098010076e-2,
            1.0073208443633202e-2,
            2.8209479177387814e-151
        ]);
        checkTensor(actual, expected, 14, false);
    });
});

describe('isPrime()', () => {
    it('should check if the input integer is a prime number', () => {
        expect(T.isPrime(2)).toBe(1);
        expect(T.isPrime(3)).toBe(1);
        expect(T.isPrime(121)).toBe(0);
    });
    it('should check if each element is a prime number', () => {
        let X = T.fromArray([
            [1, 9, 37, 113],
            [998, 16387, 4999, 2499002]
        ]);
        let actual = T.isPrime(X);
        let expected = T.fromArray([[0, 0, 1, 1], [0, 0, 1, 0]], [], T.LOGIC);
        checkTensor(actual, expected);
    });
});

describe('factor()', () => {
    it('should return the prime number itself', () => {
        checkTensor(T.factor(131), T.fromArray([131]));
        checkTensor(T.factor(4999), T.fromArray([4999]));
    });
    it('should factorize composite numbers', () => {
        checkTensor(T.factor(65536), T.tile(2, [16]));
        checkTensor(T.factor(126904023), T.fromArray([3, 3, 3, 131, 35879]));
        checkTensor(T.factor(998877665544), T.fromArray([2, 2, 2, 3, 11, 569, 6649609]));
    });
});
