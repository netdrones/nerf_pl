.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, %%2}'


install:
	conda env update -f environment.yml

train-picnic:
	gsutil -m cp -r gs://lucas.netdron.es/picnic-COLMAP .
	mv picnic-COLMAP picnic
	python generate_splits.py picnic/dense/images picnic/picnic.tsv picnic picnic/database.db
	if [ ! -d "./picnic/cache" ]; then \
	  python prepare_phototourism.py --root_dir ./picnic --img_downscale 2; \
	fi
	sh +x scripts/train_picnic.sh

colmap-picnic:
	sh +x bin/run_colmap.sh ./picnic

eval-truck:
	python generate_splits.py truck/dense/images truck/brandenburg.tsv truck truck/database.db
	mv truck brandenburg_gate
	python eval.py \
	  --root_dir brandenburg_gate \
	  --dataset_name phototourism --scene_name truck_eval \
	  --split test --N_samples 256 --N_importance 256 \
	  --N_vocab 1500 --encode_a --encode_t \
	  --ckpt_path nerf_6_29/nerfw_truck/epoch=10.ckpt \
	  --chunk 16384

train-truck: colmap-truck
	python generate_splits.py truck/dense/images truck/truck.tsv truck truck/database.db
	if [ ! -d "./truck/cache" ]; then \
	  python prepare_phototourism.py --root_dir ./truck --img_downscale 2; \
	fi
	sh +x scripts/train_truck.sh

colmap-truck: download-truck
	sh +x bin/run_colmap.sh ./truck

download-truck:
	if [ ! -d "./truck" ]; then gsutil -m cp -r gs://lucas.netdron.es/truck .; fi

eval-playground:
	mv playground brandenburg_gate
	python eval.py \
	  --root_dir brandenburg_gate \
	  --dataset_name phototourism --scene_name playground_eval \
	  --split test --N_samples 256 --N_importance 256 \
	  --N_vocab 1500 --encode_a --encode_t \
	  --ckpt_path nerf_6_29/nerfw_playground/epoch=10.ckpt \
	  --chunk 16384

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
