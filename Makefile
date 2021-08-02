include include.train.mk
include include.eval.mk

.DEFAULT_GOAL := help

install:
	conda env update -f environment.yml

build-opencv:
	sh +x bin/install_opencv.sh
