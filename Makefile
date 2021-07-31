include include.train.mk
include include.eval.mk

.DEFAULT_GOAL := help

install:
	conda env update -f environment.yml
	if [ ! -d ~/ws/git/opencv-python ]; then sh +x bin/install_opencv.sh; fi

build-opencv:
	sh +x bin/install_opencv.sh
