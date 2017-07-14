# JASMAL

**JASMAL** stands for **J**ust **A**nother Java**S**cript **MA**trix **L**ibrary,
or **JA**va**S**cript **MA**trix **L**ibrary. This is a **work-in-progress**
library I used to create interactive simulations on my
[blog](research.wmz.ninja/articles/2017/06/bartlett-mvdr-beamformer-in-the-browser.html).

Despite its name, JASMAL can actually handle multi-dimensional arrays. It
also has built-in complex number support and provides flexible indexing schemes.
It also support broadcasting for various binary operations.

# Basic Usage

To access the JASMAL engine, simply use `var T = require('jasmal').JasmalEngine.createInstance()`.

## Creation
``` JavaScript
// creates a complex number
var c = T.complexNumber(1, -1);
// creates a 3 x 3 zero matrix
var Z = T.zeros([3, 3]);
// creates a 3 x 3 identity matrix
var I = T.eye([3, 3]);
// creates a 3 x 4 x 3 array whose elements are all ones with data type INT32
var X = T.ones([3, 4, 3], T.INT32);
// from JavaScript arrays
var A = T.fromArray([[1, 2], [3, 4]]); // real
var C = T.fromArray([[1, 2], [3, 4]], [[-1, -2], [-3, -4]]); // complex
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
A.get([0, 1, 1, 2, 0, 1], ':'); // Sample rows
```

### Accessing the underlying data storage

JASMAL stores multi-dimensional arrays in the row major order. You can
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
if (A.hasComplexStorage()) {
    im = A.imagData;
    im[0] = -1;
}
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
T.sin([1, 2, 3]);
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
```

## Data functions

``` JavaScript
var A = T.fromArray([[1, 2, 3], [4, 5, 6]]);
// Gets the sum of all the elements.
var sum = T.sum(A);
// Sum each row and returns a column vector. We specify keepDims = true here.
var sums = T.sum(A, 1, true);
```