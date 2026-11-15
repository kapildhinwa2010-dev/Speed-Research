# Speed-Research

[![CI](https://github.com/kapildhinwa2010-dev/Speed-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Speed-Research/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/Speed-Research/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/Speed-Research/actions/workflows/codeql.yml)

Empirical scaling of compression runtimes: timing tables plus a fitted complexity exponent.

## What it does

Times three compression methods (SVD rank-4, 8-bit quantization, 1-D
Fourier fit) on random matrices across sizes, then fits
log(seconds) = exponent * log(size) + c per method so you can compare
the measured exponent against the theoretical one (SVD is cubic,
quantization is linear).

    from speed_research import complexity_exponent, scaling_table

    rows = scaling_table(sizes=(32, 64, 128, 256))
    print(complexity_exponent(rows, "svd_rank4"))

## API

- time_method(func, repeats) - best-of-N wall-clock seconds
- scaling_table(sizes, repeats, seed) - seconds per method per size
- complexity_exponent(rows, method) - fitted log-log slope

## Honest results and limitations

- Exponents are measured on one machine with one BLAS build; treat them
  as indicative, not certified. Cache effects and threading bend the
  curve, especially at small sizes.
- Best-of-N timing is optimistic; medians would be more conservative.
- Random square matrices only; structured inputs time differently.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
