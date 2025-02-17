.PHONY: clean html

default: clean html

clean:
	rm -rf build
	rm -rf generated
	jupyter nbconvert --clear-output --inplace **/*.ipynb

html:
	sphinx-build -b html . build
