install:
	@python setup.py install

clean:
	@rm -rf build dist .eggs *.egg-info
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +

black: clean
	@isort --profile black jmetal/ examples/
	@black jmetal/

lint:
	@mypy jmetal/ --show-error-codes

tests:
	@python -m unittest discover --quiet