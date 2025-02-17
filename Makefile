.PHONY: docs

docs:
	rm -rf docs/build
	uv run --group doc jupyter nbconvert --clear-output --inplace docs/**/*.ipynb
	uv run --group doc sphinx-build -b html docs docs/build
