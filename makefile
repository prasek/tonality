.PHONY: default
default: clean deps build

.PHONY: deps
deps: venv
	venv/bin/pip install -r requirements.txt

.PHONY: build
build: 
	source venv/bin/activate && \
	python setup.py py2app

.PHONY: clean
clean:
	rm -rf build dist

.PHONY: wipe
wipe: clean
	rm -rf venv features

.PHONY: venv
venv:
	virtualenv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install --upgrade setuptools
