all: clean check test
check: lint mypy

lint:
	ruff check clouddrift tests	

format:
	ruff format clouddrift tests

mypy:
	mypy --config-file pyproject.toml

test:
	coverage run -m unittest discover -s tests -p "*.py"

clean:
	rm -rf ~/.clouddrift
	rm -rf /tmp/clouddrift
	rm -rf ./data/*