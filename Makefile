.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, %%2}'

install:
	conda env update -f environment.yml

train-brandenburg: download-brandenburg
	if [ ! -d "./brandenburg_gate/cache" ]; then \
		python prepare_phototourism.py --root_dir ./brandenburg_gate --img_downscale 2; \
	fi
	sh +x scripts/train_brandenburg.sh

download-brandenburg:
	if [ ! -d "./brandenburg_gate" ]; then gsutil -m cp -r gs://lucas.netdron.es/brandenburg_gate .; fi

download-lego:
	if [ ! -d "./nerf_synthetic" ]; then gsutil -m cp -r gs://lucas.netdron.es/nerf_synthetic .; fi

train-lego: download-lego
	sh +x scripts/train_blender.sh

clean:
	rm -rf brandenburg_gate
	rm -rf nerf_synthetic
	rm -rf logs
