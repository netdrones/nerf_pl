include include.train.mk
include include.eval.mk

.DEFAULT_GOAL := help

install:
	conda env update -f environment.yml
	gsutil -m cp -r gs://netdron.es/opencv/*
	pip install *.whl
	rm *.whl

build-opencv:
	sh +x bin/install_opencv.sh
