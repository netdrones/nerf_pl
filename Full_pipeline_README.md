# Full NeRF PixPole Pipeline
## [**Preparing Capture**](https://github.com/netdrones/headless-camera)

> Note: this step only needs to be done once.


### Install Android Debug Bridge
on MacOS, [Brew](https://brew.sh/) can be used to install ADB (Android Debug Bridge)

```bash
brew install android-platform-tools
```

### Setting up Pixels
Enable developer mode. Phone will indicate once you have done this

`settings > About phone > tap 'Build number' 10 times`

Enable USB Debugging

`Developer options > USB debugging [on]`

Then, plug the Pixel into your computer. Running `adb devices` should display some device name if this process has been done correctly.

Download [headless-camera-2021-04-23-14-40.apk](https://drive.google.com/file/d/1ErbF8sHa4YNLyETOg5EGPiDFL9S_12aj/view)

Get the location where this was downloaded to your device and access this location via your terminal.

If it was downloaded to your Downloads folder, this can likely be accessed with `cd ~/Downloads`. Otherwise, navigate to where you downloaded the file in your system's Finder and capture the path (`option + command + c` on mac).

Then execute `cd /that/file's/path`


Uninstall prior ADB programs from the phone
```
adb uninstall es.netdron.headlesscamera
```
And install the `.apk` file you just downloaded
```
adb install headless-camera-2021-04-23-14-40.apk
```
Grant the `Headless Camera` application (what takes the photos) permission

```
adb shell pm grant es.netdron.headlesscamera android.permission.CAMERA
```
and your pixel is good to go

## **Performing Capture**
### Setting up PixelPole
The two halves of the Pixpole are twisted together. There are arrows to be lined up indicating where you should attach the pieces of the pole. The key detail is that all phone "pockets" are aligned.
### Taking photos
Start by logging the photo captures. Open up three different terminal windows. In one, run:
``` bash
adb logcat > adb-logcat.txt
```
This creates the file to store the logs

Then, in your other terminal window, run
```bash
tail -f adb-logcat.txt | egrep "es.netdron.headlesscamera.MainActivity@[0-9]+:"
```
Finally, in your third terminal, run this command to start the photo capture process.
```bash
adb shell am start -n es.netdron.headlesscamera/.MainActivity -a es.netdron.headlesscamera.START
```
You may get an error message of some kind - this doesn't mean anything. The pixel is now capturing pictures. Usually, it will start beeping, but sometimes the phone will not be beeping: it is likely still taking photos (see how to check if this is the case in Common Pitfalls section).

Unplug the current phone and plug in another one, executing just the below command. Repeat for each phone.
``` bash
adb shell am start -n es.netdron.headlesscamera/.MainActivity -a es.netdron.headlesscamera.START
```

Now that all phones are capturing images, put each in your Pixel Pole holder. It is recommended to mark each phone with a number and put the lowest number at the low position on the Pixpole, and the highest number at the highest position, allowing you to easily sort photos into the `high`, `mid`, and `low` angles.

### Using the PixPole
Walk slowly around the target building with the phone cameras always facing the building. Target being 10-20 feet away from the building, but this is not always realistic. When done, carefully lower the Pixpole, connect each phone to your computer, and run the following command
```
adb shell am start -n es.netdron.headlesscamera/.MainActivity -a es.netdron.headlesscamera.STOP
```
### Exporting photos to your computer
First, create a folder to house all photos
```
mkdir <location>
```
Then export the photos to your computer and delete them from your phone (note that `<location>` chosen must be the same in the above and below commands)
```
adb pull /storage/emulated/0/Android/media/es.netdron.headlesscamera/HeadlessCamera/ <location>
cd <location> && cd HeadlessCamera && mv * .. && cd .. && rm -rf HeadlessCamera && cd ..
adb shell rm -rf /storage/emulated/0/Android/media/es.netdron.headlesscamera/HeadlessCamera
```
Repeat these same commands for each phone
## **Uploading Images to Google Cloud**
go back out to the folder containing your `<location>` directory and upload your photos to google cloud. 
```
cd ..
gsutil -m cp -r <location> gs://sid.netdron.es
```
The formula to this command is `gsutil -m cp -r <item to upload> <place to upload it>`. It therefore also works in reverse (this is how you get the images down from google cloud to the virtual machine you are about to create)
## **Creating Virtual Machine**
create a virtual machine with the computing power to process your images.

Choose the name

``` bash
export INSTANCE_NAME="<name_of_choice>"
```

Create the instance

``` bash
gcloud beta compute instances create $INSTANCE_NAME \
	  --project netdrones \
	  --zone us-central1-a \
	  --custom-cpu 12 \
	  --custom-memory 64 \
	  --accelerator type=nvidia-tesla-v100,count=2 \
	  --maintenance-policy TERMINATE --restart-on-failure \
	  --source-machine-image nerf-machine-image
```
This will take some time.

Then log into the newly created instance
```bash
gcloud compute ssh $INSTANCE_NAME --project netdrones --zone us-central1-a
```


A common issue here is `ERROR: (gcloud.compute.ssh) [/usr/bin/ssh] exited with return code [255].` Simply run the above command again until you are presented with `welcome to Ubuntu...` (may have to run 6+ times)

## [**Processing Images**](https://github.com/netdrones/nerf_pl/tree/nerfw)
Access the nerf_pl folder and download your images from google cloud
```
cd ~/ws/git/netdrones-nerf/nerf_pl
gsutil -m cp -r gs://sid.netdron.es/<location> .
```
Activate the necessary modules (for the next steps you **must** be in the nerf_pl folder)
```
make install
conda activate nerf_pl
```
run the NeRF script on your images
```
mv <location> images
sh +x bin/train.sh -i images
```
This will take some time

## **Getting results**
Most importantly, you must upload the `ckpts` folder to google cloud - this is what contains the trained NeRF model
```bash
gsutil -m cp -r ckpts gs://sid.netdron.es/<location>
```
If you are not in the nerf_pl environment
```
make install
conda activate nerf_pl
```
Then, you can use the trained algorithm to render your images
```bash
python
from renderer import *
render = Render('ckpts/images/epoch=19.ckpt', 'images')
render.image_paths
```
this will output a list of images with a number next to each (their idx number). Choose one of these numbers and use the below command to render an image view. If you want more control in choosing what image you want to source from, go to **common pitfalls** and **remote desktop**
```
render_circle(render, <idx_number>, <radius>, <15>)
```
Then upload your results to google cloud:
```
gsutil -m cp -r results gs://sid.netdron.es/<location>
```

## **Common Pitfalls**
### **Pixels not taking pictures**

There are a lot of reasons why the Pixel might not be taking pictures. First, plug the phone in and run 
```
adb devices
```
**Phone not seeing Pixel** 

if, under `list of devices attached` there is no output, your phone is not set up properly.

First, make sure `USB-debugging` is turned on. This is detailed in the earlier **Setting up Pixels** section. If this doesn't fix it, go to developer settings and enable `PTP` USB transfer
``` bash
search > settings > search > developer options > search > USB > Default USB Configuration > Default USB Configuration > PTP
```
If this does not solve the problem, plug the phone again into the computer and run
```
adb kill-server
adb start-server
```
**Pixel not making sound**

sometimes the Pixels just don't make noise when taking photos. This can either be a non issue - the phone is taking photos, or a big issue - the phone isn't taking photos. Generally, if a phone is beeping when you start the command and then stops making noise, something has broken. Stop the photo taking process and try again. However, if the phone isn't making noise from the start, it is recommended to check that the phone is actually taking pictures.

To test this, first run the following commands (with pixel plugged into computer):
```
adb shell rm -r /storage/emulated/0/Android/media/es.netdron.headlesscamera/HeadlessCamera
```
Then follow instructions detailed in **Taking photos** to take some pictures, and stop the capture:
```
adb shell am start -n es.netdron.headlesscamera/.MainActivity -a es.netdron.headlesscamera.STOP
```
Try this command:
```
adb shell ls /storage/emulated/0/Android/media/es.netdron.headlesscamera/HeadlessCamera/
```
If you get a list of photos, you just have a quiet phone. If you don't, you have a broken phone. Restart the phone and try the troubleshooting steps listed in **Phone not seeing Pixel**
### **No directory /Users/name/ws/git/netdrones-nerf/nerf_pl/users/name/ws/git/netdrones-nerf...**
This is the error you will run into if you run the NeRF script without first running 
```
make install
conda activate nerf_pl
```
You must now remove the cache folder and again run the commands to start the NeRF training
``` 
cd ~/ws/git/netdrones-nerf/nerf_pl/images && rm -rf cache 
make install
conda activate nerf_pl
sh +x bin/train.sh -i images
```
### **Pictures outputted are a single color**
This is a frustrating one - computer renders images sometimes for hours and then they end up being of poor quality. I recommend just running the commands in **Getting results** again, but choosing a different idx photo. However, there is another solution.

**Remote Desktop**

Go to [this link](https://remotedesktop.google.com/headless). Log into your Netdron.es Google account. Click through the steps until you get to the screen `Set up Another Computer`. Copy the code from the Linux section, and input this on your terminal (assuming you are logged into your virtual machine). Choose your pin.

Then, go to [this page](https://remotedesktop.google.com/access/), again logging into your Google Account, and if all has worked, you should see a `name of your choice` instance to click into - click this and input the pin.

You now have remote access to the desktop of this virtual machine. Now, you can run the below commands again:
```bash
python
from renderer import *
render = Render('ckpts/images/epoch=19.ckpt', 'images')
render.image_paths
```
However, you are now able to run 
```
png_from_idx(render, <idx_of_image>)
```
This will produce a .JPG file in the nerf_pl folder that shows where `render_circle` will start from. Execute this command for as many of the idx's as you would like, and you can view them to decide which one you want to start from. You will find these images at `ws > git > netdrones-nerf > nerf_pl` in the remote desktop computer.
### Too many virtual machines in a given region
There is only so much computing power in each region. If you are getting an error related to this, try switching the area (from `east1-a` to `west1-a`). You will also need to change the following command to log into the VM (replacing the same things).