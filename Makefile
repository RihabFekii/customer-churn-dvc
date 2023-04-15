PYTHON_INTERPRETER = python3

env:
	@echo ">>> Creating a python virtual environment with venv"
	$(PYTHON_INTERPRETER) -m venv my-env
	@echo ">>> A new virtual env is created. Activate it with:\nsource my-env/bin/activate ."
