import BenchMark = require('benchmark');
import { JasmalEngine } from '../';
const suite = new BenchMark.Suite();
const T = JasmalEngine.createInstance();

// Indexing.
(() => {
    let n = 20;
    let nn = n * n;
    let arr = new Float64Array(n * n * n);
    let A = T.zeros([n, n, n]);
    suite.add('Native indexing set', () => {
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                for (let k = 0;k < n;k++) {
                    arr[i * nn + j * n + k] = i + j + k;
                }
            }
        }
    });
    suite.add('Native indexing get', () => {
        let acc = 0;
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                for (let k = 0;k < n;k++) {
                    acc = arr[i * nn + j * n + k];
                }
            }
        }
    });
    suite.add('Tensor.realData set', () => {
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                for (let k = 0;k < n;k++) {
                    A.realData[i * nn + j * n + k] = i + j + k;
                }
            }
        }
    });
    suite.add('Tensor.realData get', () => {
        let acc = 0;
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                for (let k = 0;k < n;k++) {
                    acc = A.realData[i * nn + j * n + k];
                }
            }
        }
    });
    suite.add('Tensor.setEl() flat', () => {
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                for (let k = 0;k < n;k++) {
                    A.setEl(i * nn + j * n + k, i + j + k);
                }
            }
        }
    });
    suite.add('Tensor.getEl() flat', () => {
        let acc = 0;
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                for (let k = 0;k < n;k++) {
                    acc = <number>A.getEl(i * nn + j * n + k);
                }
            }
        }
    });
    suite.add('Tensor.setEl()', () => {
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                for (let k = 0;k < n;k++) {
                    A.setEl(i, j, k, i + j + k);
                }
            }
        }
    });
    suite.add('Tensor.getEl()', () => {
        let acc = 0;
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                for (let k = 0;k < n;k++) {
                    acc = <number>A.getEl(i, j, k);
                }
            }
        }
    });
    suite.add('Tensor.set() flat', () => {
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                for (let k = 0;k < n;k++) {
                    A.set(i * nn + j * n + k, i + j + k);
                }
            }
        }
    });
    suite.add('Tensor.get() flat', () => {
        let acc = 0;
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                for (let k = 0;k < n;k++) {
                    acc = <number>A.get(i * nn + j * n + k);
                }
            }
        }
    });
    suite.add('Tensor.set()', () => {
    for (let i = 0;i < n;i++) {
        for (let j = 0;j < n;j++) {
            for (let k = 0;k < n;k++) {
                A.set(i, j, k, i + j + k);
            }
        }
    }
    });
    suite.add('Tensor.get()', () => {
        let acc = 0;
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                for (let k = 0;k < n;k++) {
                    acc = <number>A.get(i, j, k);
                }
            }
        }
    });
})();

// Set batch
(() => {
    let n = 100;
    let arr = new Float64Array(n * n);
    let A = T.zeros([n, n]);
    suite.add('Native set element to one', () => {
        for (let i = 0;i < n;i++) {
            for (let j = 0;j < n;j++) {
                arr[i * n + j] = 1;
            }
        }
    });
    suite.add('Tensor.set(\':\', 1)', () => {
        A.set(':', 1);
    });
})();

// Get batch
(() => {
    let n = 100;
    let arr = new Float64Array(n * n);
    let A = T.zeros([n, n]);
    suite.add('Native get elements at even indices', () => {
        let arrOut = new Float64Array(Math.floor(n * n / 2));
        for (let i = 0, j = 0;i < n * n;i += 2, j++) {
            arrOut[j] = arr[i];
        }
    });
    suite.add('Tensor.get(\'::2\')', () => {
        A.get('::2');
    });
})();

suite.on('cycle', e => {
    console.log(String(e.target));
}).run();
