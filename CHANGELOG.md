### v0.1.4
* Implemented FFT KDE to evaluate PDF by default. 

### v0.1.3
* Corrected bug that left out extreme value of x in pdf.
* Warning for the use of q_range.
* Changed to KL divergence to compare histogram and theory in test of quartic potential, instead of mean square error that gives large errors for outliers with small probability (overestimate probability)


### v0.1.2:
* Removed unused dependencies (pandas, jupyter, jupyterlab).
* Compatible with python < 3.13
* Published to PyPI

### v0.1.1
Update to use scipy simpson, fallback to simps for older versions. Update to
python < 3.12.

### v0.1.0

Langesim version 0.1.0
Langevin simulator of an overdamped brownian particle in an arbitrary time-dependent potential.

See the documentation at:

https://gabrieltellez.github.io/langesim/

First published version to PyPI.