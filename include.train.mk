train-house: download-house
	sh +x bin/train.sh house house_scale1_nerfw 2

train-playground: download-playground
	sh +x bin/train.sh playground playground_scale1_nerfw 2

train-truck: download-truck
	sh +x bin/train.sh truck truck_scale1_nerfw 2

train-brandenburg: download-brandenburg
	sh +x bin/train.sh brandenburg_gate brandenburg_scale2_nerfw 2

download-house:
	if [ ! -d "house" ]; then gsutil -m cp -r gs://lucas.netdron.es/house .; fi

download-playground:
	if [ ! -d "playground" ]; then gsutil -m cp -r gs://lucas.netdron.es/playground .; fi

download-truck:
	if [ ! -d "truck" ]; then gsutil -m cp -r gs://lucas.netdron.es/truck .; fi

download-brandenburg:
	if [ ! -d "brandenburg_gate" ]; then gsutil -m cp -r gs://lucas.netdron.es/brandenburg_gate .; fi
