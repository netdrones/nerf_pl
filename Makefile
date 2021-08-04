include include.train.mk
include include.eval.mk

.DEFAULT_GOAL := help

install:
	conda env update -f environment.yml

install-custom-opencv:
	pip install --upgrade pip
	if [ ! -d bin/opencv ]; then gsutil -m cp -r gs://netdron.es/opencv bin; fi
	pip install bin/opencv/*.whl

build-opencv:
	sh +x bin/install_opencv.sh
