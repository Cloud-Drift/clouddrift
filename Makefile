all:  format check test
build-deps:
	micromamba install -n clouddrift build twine docutils
library-deps:
	micromamba create -n clouddrift -f environment.yml
dev-deps:
	micromamba install -n clouddrift ruff mypy
test-deps:
	micromamba install -n clouddrift coverage matplotlib-base cartopy
all-deps: build-deps library-deps dev-deps test-deps

build: build-deps
	python -m build
install-local: build
	micromamba install dist/clouddrift*.whl
dev: library-deps dev-deps test-deps
check: lint mypy
lint:
	ruff check clouddrift tests	

format:
	ruff format clouddrift tests

mypy:
	mypy --install-types --config-file pyproject.toml

test:
	coverage run -m unittest discover -s tests -p "*.py"


clean:
	rm -rf ~/.clouddrift
	rm -rf /tmp/clouddrift
	rm -rf ./data/*