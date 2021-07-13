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

eval-brandenburg: download-ckpts
	python eval.py \
	  --root_dir brandenburg_gate/ \
	  --dataset_name phototourism --scene_name brandenburg_test \
	  --split test --N_samples 256 --N_importance 256 \
	  --N_vocab 1500 --encode_a --encode_t \
	  --ckpt_path ckpts/brandenburg_scale8_nerfw/epoch=6.ckpt \
	  --chunk 16384

download-ckpts:
	if [ ! -d ckpts/ ]; then gsutil -m cp -r gs://lucas.netdron.es/nerfw_ckpts/ .; fi
	mv nerfw_ckpts ckpts

download-house-ckpts:
	if [ ! -d "./house_scale2_nerfw" ]; then gsutil -m cp -r gs://lucas.netdron.es/house_scale2_nerfw .; fi

download-test-ckpts:
	if [ ! -d "./nerf_6_29" ]; then gsutil -m cp -r gs://lucas.netdron.es/nerf_6_29 .; fi
