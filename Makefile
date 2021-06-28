.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, %%2}'

install:
	conda env update -f environment.yml

train-playground:
	python generate_splits.py playground/dense/images playground/playground.tsv playground playground/database.db
	if [ ! -d "./playground/cache" ]; then \
	  python prepare_phototourism.py --root_dir ./playground --img_downscale 2; \
	fi
	sh +x scripts/train_playground.sh

colmap-playground: download-playground
	sh +x bin/run_colmap.sh ./playground

download-playground:
	if [ ! -d "./playground" ]; then gsutil -m cp -r gs://lucas.netdron.es/data/playground .; fi

eval-brandenburg: download-ckpts
	python eval.py \
	  --root_dir brandenburg_gate/ \
	  --dataset_name phototourism --scene_name brandenburg_test \
	  --split test --N_samples 256 --N_importance 256 \
	  --N_vocab 1500 --encode_a --encode_t \
	  --ckpt_path ckpts/brandenburg_scale8_nerfw/epoch=6.ckpt \
	  --chunk 16384

download-ckpts:
	if [ ! -d ckpts/ ]; then gsutil -m cp -r gs://lucas.netdrones/nerfw_ckpts/ .; fi
	mv nerfw_ckpts ckpts

train-brandenburg: download-brandenburg
	wget https://nerf-w.github.io/data/selected_images/brandenburg.tsv
	mv brandenburg.tsv brandenburg_gate/
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
