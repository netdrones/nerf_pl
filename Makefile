.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, %%2}'


install:
	conda env update -f environment.yml

eval-house: download-house-ckpts
	mv house brandenburg_gate
	python eval.py \
	  --root_dir brandenburg_gate \
	  --dataset_name phototourism --scene_name house_eval \
	  --split test --N_samples 2356 --N_importance 256 \
	  --N_vocab 1500 --encode_a --encode_t \
	  --ckpt_path house_scale2_nerfw/epoch=19.ckpt \
	  --chunk 16384
	mv brandenburg_gate house

train-house: colmap-house
	sh +x bin/train.sh ./house house_scale2_nerfw 2

colmap-house: download-house
	if [ ! -d ".house/dense" ]; then sh _x bin/run_colmap.sh ./house

download-house:
	if [ ! -d "./house" ]; then gsutil -m cp -r gs://lucas.netdron.es/house .; fi

download-house-ckpts:
	if [ ! -d "./house_scale2_nerfw" ]; then \
		gsutil -m cp -r gs://lucas.netdron.es/house_scale2_nerfw .;
	fi

eval-playground: download-test-ckpts
	mv playground brandenburg_gate
	python eval.py \
	  --root_dir brandenburg_gate \
	  --dataset_name phototourism --scene_name house_eval \
	  --split test --N_samples 2356 --N_importance 256 \
	  --N_vocab 1500 --encode_a --encode_t \
	  --ckpt_path nerf_6_29/nerfw_playground/epoch=10.ckpt \
	  --chunk 16384
	mv playground house

train-playground: colmap-playground
	sh +x bin/train.sh ./playground playground_scale2_nerfw 2

colmap-playground: download-playground
	if [ ! -d ".playground/dense" ]; then sh +x bin/run_colmap.sh ./playground

download-playground:
	if [ ! -d "./playground" ]; then gsutil -m cp -r gs://lucas.netdron.es/playground .; fi


eval-truck: download-test-ckpts
	mv truck brandenburg_gate
	python eval.py \
	  --root_dir brandenburg_gate \
	  --dataset_name phototourism --scene_name truck_eval \
	  --split test --N_samples 256 --N_importance 256 \
	  --N_vocab 1500 --encode_a --encode_t \
	  --ckpt_path nerf_6_29/nerfw_truck/epoch=10.ckpt \
	  --chunk 16384
	mv brandenburg_gate truck

train-truck: colmap-truck
	sh +x scripts/train_truck.sh ./truck truck_scale2_nerfw 2

colmap-truck: download-truck
	sh +x bin/run_colmap.sh ./truck

download-truck:
	if [ ! -d "./truck" ]; then gsutil -m cp -r gs://lucas.netdron.es/truck .; fi

download-test-ckpts:
	if [ ! -d "./nerf_6_29" ]; then gsutil -m cp -r gs://lucas.netdron.es/nerf_6_29 .; fi

eval-brandenburg: download-ckpts
	python eval.py \
	  --root_dir brandenburg_gate/ \
	  --dataset_name phototourism --scene_name brandenburg_test \
	  --split test --N_samples 256 --N_importance 256 \
	  --N_vocab 1500 --encode_a --encode_t \
	  --ckpt_path ckpts/brandenburg_scale8_nerfw/epoch=6.ckpt \
	  --chunk 16384

train-brandenburg: download-brandenburg
	wget https://nerf-w.github.io/data/selected_images/brandenburg.tsv
	mv brandenburg.tsv brandenburg_gate/
	sh +x scripts/train.sh ./brandenburg_gate brandenburg_scale2_nerfw 2

download-brandenburg:
	if [ ! -d "./brandenburg_gate" ]; then gsutil -m cp -r gs://lucas.netdron.es/brandenburg_gate .; fi

download-ckpts:
	if [ ! -d ckpts/ ]; then gsutil -m cp -r gs://lucas.netdrones/nerfw_ckpts/ .; fi
	mv nerfw_ckpts ckpts

