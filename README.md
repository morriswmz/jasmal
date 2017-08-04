# JASMAL

**JASMAL** stands for **J**ust **A**nother Java**S**cript **MA**trix **L**ibrary,
or **JA**va**S**cript **MA**trix **L**ibrary. This is a **work-in-progress**
library I used to create interactive simulations on my
[blog](research.wmz.ninja/articles/2017/07/music-in-the-browser.html).

Despite its name, JASMAL can actually handle multi-dimensional arrays. It also has

* built-in complex number support
* flexible indexing schemes (e.g., `get('::-1', ':')` returns a new matrix
  with the rows reversed)
* broadcasting support for various binary operations
* subroutines for common matrix operations such as `trace()`, `inv()`, `det()`,
  `linsolve()`, `rank()`, `kron()`.
* subroutines for LU decomposition, QR decomposition, singular value
  decomposition, and eigendecomposition for both real and complex matrices 

# Basic Usage

To access the JASMAL engine, simply use `var T = require('jasmal').JasmalEngine.createInstance()`.

## Type Basics

JASMAL is built around [tensor](src/lib/tensor.ts) objects, which use typed
arrays for data storage. In plain JavaScript, jagged arrays is usually used to
store multi-dimensional arrays. The following code show how to convert between
JavaScript arrays and tensor objects:

``` JavaScript
// Creates a tensor from JavaScript arrays.
let A = T.fromArray([[1, 2], [3, 4]]); // real 2 x 2
let C = T.fromArray([[1, 2], [3, 4]], [[-1, -2], [-3, -4]]); // complex 2 x 2
// Convert a tensor to a JavaScript array.
let a = A.toArray(true); // real part only, arr = [[1, 2], [3, 4]]
let [reC, imC] = C.toArray(false); // convert both real and imaginary parts
                                   // reC = [[1, 2], [3, 4]];
                                   // imC = [[-1, -2], [-3, -4]];

```

Note that during the conversions the data are **always copied** because JASMAL
cannot be sure whether you will modify the array element in the future.

JASMAL also includes a built-in [ComplexNumber](src/lib/complexNumber.ts)
type to support complex scalars. Complex numbers can be created with the
following code:

``` JavaScript
// Creates a complex number.
let c = T.complexNumber(1, -1);
// Retrieves the real part.
let re = c.re;
// Test if c is a ComplexNumber instance.
console.log(T.isComplexNumber(c)); // true
```

Instead of only allowing tensor objects as inputs, most of the JASMAL functions
also allows JavaScript arrays, ComplexNumber instances, or numbers as
inputs. Conversion to tensor objects is automatically performed internally.
For instance, `T.add([[1], [2]], [[3, 4]])` produces the same result as the 
following code:

``` JavaScript
let x = T.fromArray([[1], [2]]);
let y = T.fromArray([[3, 4]]);
let z = T.add(x, y);
```

This is very convenient, but it leads to one problem: how is the output type
determined? For instance, should `T.add(1, 2)` output a number or a tensor
object? In JASMAL, unless otherwise specified, the following rules applies:

* If all the inputs are scalars (`number` or `ComplexNumber`), the output will
  be a scalar. In this case, if the output's imaginary part is zero, a
  JavaScript number will be returned. Otherwise a `ComplexNumber` instance will
  be returned.
* If any of the inputs is a tensor object or a JavaScript array, the output will
  be a tensor object.
* Some functions have a parameter named `keepDims`. If `keepDims` is set to
  `true`, the output will always be a tensor object.

## Creation

``` JavaScript
// Creates a 3 x 3 zero matrix.
var Z = T.zeros([3, 3]);
// Creates a 3 x 3 identity matrix.
var I = T.eye([3, 3]);
// Creates a 3 x 4 x 3 array whose elements are all ones with data type INT32.
var X = T.ones([3, 4, 3], T.INT32);

```

## Indexing
``` JavaScript
var A = T.eye([3, 3]);
A.set(0, 10); // sets the first element to 10
A.set(':', 1); // sets all the elements to 1
A.set(-1, 10); // sets the last element to 10
A.set(0, ':', 2); // sets all the elements in the first row to 2
A.set('::2', 0); // sets all the elements with even indices to 0
A.set([0, -1], [0, -1], [[1, 2], [3, 4]]); // sets four corners to 1, 2, 3, 4
A.set(function (x) { return x < 0; }, 0); // sets all negative elements to 0

A.get(0); // gets the first element
A.get(':', 0); // gets the first column as a 1D vector
// By default, get() will attempt to remove singleton dimensions. If you do not
// want this behavior, you can specify keepDims = true.
A.get(':', 0, true); // gets the first columns as a 2D column vector

A.get('::-1', ':'); // gets a new matrix with the rows reversed
A.get([0, -1], [0, -1]); // gets a new matrix consists of the four corners
A.get([0, 1, 1, 2, 0, 1], ':'); // sample rows
```

## Matrix/Tensor manipulation

``` JavaScript
var A = T.fromArray([[1, 2, 3], [4, 5, 6]]);
// Reshaping.
var B = T.reshape(A, [3, 2]);
// The following is equivalent to the above, where we use -1 to indicate that
// the length of this dimension need to be calculated automatically. At most one
// -1 can be used in the new shape.
var B1 = T.reshape(A, [-1, 2]);
// Flattening
var a = T.flatten(A); // should have shape [6]
// Vectorizing
var v = T.vec(A); // should have shape [6, 1]
// Remove singleton dimensions.
var S = T.ones([1, 2, 1, 3]);
T.squeeze(S); // should have shape [2, 3]
// Concatenating at the specified dimension.
var X1 = T.ones([2, 4, 3]);
var X2 = T.zeros([2, 2, 3]);
var X3 = T.ones([2, 1, 3]);
var Z = T.concat([X1, X2, X3], 1); // should have shape [2, 7, 3]
/* Tiling. C should be
 *  [[1, 2, 1, 2],
 *   [3, 4, 3, 4],
 *   [1, 2, 1, 2],
 *   [3, 4, 3, 4]] */
var C = T.tile([[1, 2], [3, 4]], [2, 2]);
```

## Arithmetic operations

``` JavaScript
// Only when both operands are scalar, a scalar is returned.
var s = T.add(1, 2); // returns 3

var A = T.ones(3);
// Subtract one from matrix A and return the result as a new matrix.
var B = T.sub(A, 1);
// Subtract one from matrix A but do it in-place.
T.sub(A, 1, true);

// Broadcasting
var X = T.rand([3, 3]);
var C = T.mul([[1], [2], [3]], X);
// The above operation is equivalent to the following one:
var C1 = T.matmul(T.diag([1, 2, 3]), X);
// or the following one:
var C2 = T.mul(T.tile([[1], [2], [3]], [1, 3]), X);
```

## Math functions

JASMAL supports various math functions. Many of them also accepts complex
inputs.
``` JavaScript
// Absolute value
T.abs([[1, 2], [-3, 4]]);
// Sine
T.sin(T.linspace(0, Math.PI*2, 100));
// Square root
T.sqrt(-1);
```

## Random number generation

To support seeding, Jasmal uses the [Mersenne twister](https://en.wikipedia.org/wiki/Mersenne_Twister)
engine by default to generate random numbers.

``` JavaScript
// Specify the seed.
T.seed(42);
// Retrieves an double within (0,1) with 53-bit precision.
var x = T.rand(); 
// Creates a 3x4x5 tensor whose elements sampled from the normal distribution.
var N = T.randn([3, 4, 5]);
// Generate 10 random integers within [0, 10].
var Z = T.randi(0, 10, [10]);
``` 

## Matrix operations
``` JavaScript
var A = T.rand([3, 3]), B = T.rand([3, 3]);
// Construct a complex matrix by combining real and imaginary parts.
var C = T.complex(A, B);
// Matrix multiplication
var AB = T.matmul(A, B);
// You can specify a modifier for B
var ABt = T.matmul(A, B, T.MM_TRANSPOSED);
// Extract diagonal elements.
var d = T.diag(A);
// Construct a diagonal matrix.
var D = diag(d);
// Matrix transpose/Hermitian.
var At = T.transpose(A);
var Ch = T.hermitian(C);
// Kronecker product.
var K = T.kron(A, B);
// Inverse (JASMAL uses LUP decomposition to compute the inverse)
var Ainv = T.inv(A);
// Determinant (JASMAL uses LUP decomposition to compute the determinant)
var detA = T.det(A);
// SVD
var [U1, S1, V1] = T.svd(A);
// SVD also works for complex matrices.
var [U2, S2, V2] = T.svd(C);
// Eigendecomposition for general square matrices.
var [E1, L1] = T.eig(A);
// Eigendecomposition also works for general complex square matrices.
var [E2, L2] = T.eig(C);
```

## Data functions

``` JavaScript
var A = T.fromArray([[1, 2, 3], [4, 5, 6]]);
// Gets the sum of all the elements.
var sum = T.sum(A);
// Sum each row and returns a column vector. We specify keepDims = true here.
var sums = T.sum(A, 1, true);
```

# Performance

Unfortunately, JavaScript is slow for numerical computations. Therefore,
multiply two 1000 x 1000 matrices or performing the singular value decomposition
of a 500 x 500 matrix take seconds in the browser.

Element-wise indexing can also be slow with JASMAL because of the overhead
needed for implementing the flexible indexing schemes. For instance, if you
want to set all the elements in a matrix to zero, `M.set(':', 0)` is much much
faster than the following code:
``` JavaScript
for (let i = 0;i < m;i++) {
    for (let j = 0;j < n;j++) {
        M.set(i, j, 0);
    }
}
```

### Accessing the underlying data storage directly

JASMAL stores multi-dimensional arrays in the row major order. If you want to 
completely bypass the indexing overhead of JASMAL's indexing functions, you can
directly access the underlying storage and manipulate them:

``` JavaScript
// If you want to write to the underlying storage directly, ensure that it is
// not shared.
A.ensureUnsharedLocalStorage();
var re = A.realData;
re[0] = 1;
var im;
// If a matrix does not have complex data storage, accessing its imaginary data
// storage will result in an error.
if (!A.hasComplexStorage()) {
    // Make sure the complex data storage is available.
    A.ensureComplexStorage();
}
im = A.imagData;
im[0] = -1;
```

# License

JASMAL is released under the [MIT](LICENSE.md) license.
[eigen.ts](src/lib/ops/matrix/decomp/eigen.ts)
contains several subroutines ported from the pristine Fortran code
[EISPACK](http://www.netlib.org/eispack/),
which are distributed using the Modified BSD or MIT license
([source](http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg01379.html)).
All rights reserved by the authors of EISPACK. 
