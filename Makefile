init:
	python3 -m venv my_env
	source my_env/bin/activate
	pip3 install -r requirements_dev.txt

test:
	py.test tests
