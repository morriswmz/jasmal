# Implementation Details

* Different groups of operations are organized in different sub folders.
* The `generator` folder contains tools to dynamically generator unary and binary
  operations based on pre-defined function templates.
* Functions implemented by each provider should not depend on `this` as they
  will be copied to `Jasmal` object. Therefore, in the current implementation,
  all provider interfaces are implemented using object literals.
* Unless otherwise specified, operations will not be in-place and new tensors
  will be returned.

## Empty Tensor Handling

When implementing broadcasting and reduction operations, we used recursion. If
the input tensor object is empty, either the last element of its shape array,
or some element other than the last element of its shape array, is zero.

1. If the last element of its shape array is zero, the for loop in the deepest
   level of recursion won't execute.
2. If some element other than the last element of its shape array is zero,
   then the recursion won't go deeper because the for loop responsible for
   recursive calling won't execute.

Therefore, the current implementation should handle empty tensor objects without
problems. The same applies to `tile()` and `permuteAxis()`.