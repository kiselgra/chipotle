# Chipotle
Image processing DSL implemented in C-Mera.

The DSL is described in our ELS'16 paper "A High-Performance Image Processing DSL for Heterogeneous Architectures".

## Dependencies
* C-Mera 1.0 (not c-mera-2015)
* CM-FOP

Additionally, the example code requires the following libraries to run properly:
* Cuda
* MagickWand


## Recommendation
Due to long compilation times with SBCL, we recommend Clozure-CL for Chipotle.
To do so, configure and reinstall C-Mera with the following flag: "--with-ccl".

## Test-Run & Installation
Now you can make Chipotle known to your lisp environment, e.g. by

	$ ln -s /path/to/chipotle ~/quicklisp/local-projects/chipotle

With Chipotle known to your system you can run our example filters:
	
	$ cd chipotle/examples
	$ make

## Third Party Material
Chipotle comes with the following third-party material:
* [avx_mathfun.h](http://software-lisc.fbk.eu/avx_mathfun/)
* [sse_mathfun.h](http://gruntthepeon.free.fr/ssemath/)
