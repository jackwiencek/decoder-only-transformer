# scratch/

Throwaway experiments. Not part of the `tinygpt` package, not imported by it,
not held to the same standard.

This is where we poke at tensors to build intuition — printing shapes, checking
that a hand-rolled attention matches `F.scaled_dot_product_attention`, watching
what a softmax does as head size grows. Files here are allowed to be messy and
are allowed to be deleted.

Anything that earns a permanent place moves into `src/tinygpt/` with a test.
