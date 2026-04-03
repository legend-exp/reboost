# Developers guide

## Testing

### Numba-JIT functions

Functions decorated with `@numba.njit` are compiled to native code at runtime,
making their Python source invisible to coverage tools. To ensure both the JIT
path and the pure-Python path are exercised, use the `compare_numba_vs_python`
fixture provided in `tests/conftest.py`.

The fixture calls the function twice — once through the JIT-compiled path and
once via `.py_func` — asserts that the results are numerically equal, and
returns the JIT result so it can be used directly in assertions. Equality
checking uses `np.testing.assert_allclose`, so all function outputs must be
numpy arrays or array-like values that can be converted with `np.asarray`
(e.g. scalars, lists, or awkward arrays). Pass `check_equal=False` for
functions with non-array outputs:

```python
def test_my_numba_func(compare_numba_vs_python):
    result = compare_numba_vs_python(my_module._my_njit_func, arg1, arg2)
    assert result == expected


# skip automatic equality check for non-array outputs
def test_my_numba_func_no_check(compare_numba_vs_python):
    result = compare_numba_vs_python(
        my_module._my_njit_func, arg1, arg2, check_equal=False
    )
    assert result == expected
```

For functions that take internal numba-typed data structures that cannot be
constructed outside a JIT context (e.g. the stepwise-deposition iteration
kernels in `reboost.optmap.convolve`), test the pure-Python path by
inline-patching with `monkeypatch`:

```python
def test_stochastic_kernel(monkeypatch):
    out_jit = public_api(inputs)  # exercises JIT path

    monkeypatch.setattr(module, "_kernel", module._kernel.py_func)
    out_py = public_api(inputs)  # exercises Python path
    assert ak.num(out_py).tolist() == ak.num(out_jit).tolist()
```
