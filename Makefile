.PHONY: install run test clean train

install:
	pip install -r requirements.txt

run:
	flask run

train:
	python scripts/train_model.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/

test:
	python -m pytest tests/

setup: install
	mkdir -p data/schedules data/zyte_data
	mkdir -p app/static/images