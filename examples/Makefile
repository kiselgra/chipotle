all: chp-night chp-laplace test

chp-night:
	$(MAKE) -C src chp-night
	ln -s ./src/chp-night ./chp-night

chp-laplace:
	$(MAKE) -C src chp-laplace
	ln -s ./src/chp-laplace ./chp-laplace

test:
	./chp-night 
	./chp-laplace

clean:
	$(MAKE) -C src clean
	unlink ./chp-night 2> /dev/null || true
	unlink ./chp-laplace 2> /dev/null || true
	rm night*.jpg 2> /dev/null || true
	rm laplace*.jpg 2> /dev/null || true
