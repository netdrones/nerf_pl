#!/bin/bash

SCRIPTNAME="$(basename $0)"
CLEAN=false
OVERLAP_HIGH=0.98

OPTIND=1
while getopts "hci:" opt; do
  	case "$opt" in
	h)
	  	echo "$SCRIPTNAME: flags: -h help, -c clean, -i input_dir, -o overlap_threshold"
		exit 0
		;;
	c)
	  	export CLEAN=true
		;;
	i)
	  	export WORKSPACE_DIR="$OPTARG"
		;;
	o)
	  	export OVERLAP_HIGH="$OPTARG"

	esac
done

shift "$(( OPTIND -1 ))"
if [ -z "$WORKSPACE_DIR" ]; then
  	echo "$SCRIPTNAME: Missing -i argument"
	exit 1
fi

EXP_NAME=$WORKSPACE_DIR
DOWNSCALE=2
NUM_GPUS="$(nvidia-smi --query_gpu=name --format=csv,nohearder | wc -l)"

if [ ! -d "$WORKSPACE_DIR/dense" ]
then
	if [ ! -d $WORKSPACE_DIR/images ]
   	then
		if $CLEAN
		then
		  	mkdir -p $WORKSPACE_DIR/images_cleaned
			python image_utils.py $WORKSPACE_DIR $WORKSPACE_DIR/images_cleaned $OVERLAP_HIGH
			mv $WORKSPACE_DIR/images_cleaned $WORKSPACE_DIR/images
		else
		  	mkdir -p $WORKSPACE_DIR/images
			mv $WORKSPACE_DIR/*.jpg $WORKSPACE_DIR/images 2> /dev/null
			mv $WORKSPACE_DIR/*.jpeg $WORKSPACE_DIR/images 2> /dev/null
			mv $WORKSPACE_DIR/*.png $WORKSPACE_DIR/images 2> /dev/null
			mv $WORKSPACE_DIR/*.JPG $WORKSPACE_DIR/images 2> /dev/null
			mv $WORKSPACE_DIR/*.JPEG $WORKSPACE_DIR/images 2> /dev/null
			mv $WORKSPACE_DIR/*.PNG $WORKSPACE_DIR/images 2> /dev/null

		fi
	else
	  	if $CLEAN
		then
		  	mkdir -p $WORKSPACE_DIR/images_cleaned
		  	python image_utils.py $WORKSPACE_DIR/images $WORKSPACE_DIR/images_cleaned $OVERLAP_HIGH
			rm -r $WORKSPACE_DIR/images
			mv $WORKSPACE_DIR/images_cleaned $WORKSPACE_DIR/images
		else
		  	mkdir -p $WORKSPACE_DIR/images
		  	mv $WORKSPACE_DIR/*.jpg $WORKSPACE_DIR/images 2> /dev/null
			mv $WORKSPACE_DIR/*.jpeg $WORKSPACE_DIR/images 2> /dev/null
			mv $WORKSPACE_DIR/*.png $WORKSPACE_DIR/images 2> /dev/null
			mv $WORKSPACE_DIR/*.JPG $WORKSPACE_DIR/images 2> /dev/null
			mv $WORKSPACE_DIR/*.JPEG $WORKSPACE_DIR/images 2> /dev/null
			mv $WORKSPACE_DIR/*.PNG $WORKSPACE_DIR/images 2> /dev/null

		fi
   	fi
   	sh +x bin/run_colmap.sh $WORKSPACE_DIR
fi

python generate_splits.py \
  	$WORKSPACE_DIR/dense/images \
	$WORKSPACE_DIR/$WORKSPACE_DIR.tsv \
	$WORKSPACE_DIR $WORKSPACE_DIR/database.db

if [ ! -d "$WORKSPACE_DIR/cache" ]; then \
	python prepare_phototourism.py --root_dir $WORKSPACE_DIR --img_downscale $DOWNSCALE; \
fi

python train.py \
	--root_dir $WORKSPACE_DIR --dataset_name phototourism \
    	--img_downscale $DOWNSCALE --use_cache --N_importance 64 --N_samples 64 \
    	--encode_a --encode_t --beta_min 0.03 --N_vocab 1500 \
    	--num_epochs 20 --num_gpus $NUM_GPUS --batch_size 1024 \
    	--optimizer adam --lr 5e-4 --lr_scheduler cosine \
    	--exp_name $EXP_NAME
