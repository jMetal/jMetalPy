install:
	@python setup.py build install

install-dependencies:
	@python -m pip install -e .[all]

clean:
	@rm -rf build dist .eggs *.egg-info
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +

black: clean
	@isort --profile black jmetal/ examples/
	@black jmetal/ examples/

lint:
	@mypy jmetal/ examples/ --show-error-codes

tests:
	@python -m unittest discover -q