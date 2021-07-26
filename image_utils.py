import os
import sys
import cv2 as cv
import numpy as np
import skgeom as sg
import matplotlib.pyplot as plt

from tqdm import tqdm
from exif import Image as Exif

# SIFT parameters
MIN_MATCHES = 200
FLANN_INDEX_KDTREE = 1
OVERLAP_THRESHOLD = 0.5

# Light parameters
GAMMA = 1.5
BLUR_THRESHOLD = 80
BRIGHTNESS_THRESHOLD = -8.0

def calculate_overlap(img_1, img_2):

    def area(pts):
        return sg.Polygon(np.squeeze(pts)).area()

    # Calculate homography between images and apply to corners
    H = calculate_homography(img_1, img_2)
    h, w = img_1.shape
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts, H)

    orig_area = area(pts)
    overlapped_area = area(dst)
    if overlapped_area == 0:
        raise ValueError("No overlap")

    return orig_area/overlapped_area

def calculate_homography(img_1, img_2):

    sift = cv.SIFT_create()

    # Extract features from each image
    kp_1, des_1 = sift.detectAndCompute(img_1, None)
    kp_2, des_2 = sift.detectAndCompute(img_2, None)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Generate matches
    flann = cv.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des_1, des_2, k=2)
    except Exception:
        raise ValueError("OpenCV KNN error")
        return

    # Prune matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Use matches to find the homography between img_1 and img_2
    if len(good_matches) > MIN_MATCHES:
        src_pts = np.float32([ kp_1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        return H

    else:
        raise ValueError(f"Insufficient number of matches ({len(good_matches)}/{MIN_MATCHES})")
        return

class ImageDataset:

    def __init__(self, dataset_path, min_matches=MIN_MATCHES):

        self.blur_threshold = BLUR_THRESHOLD
        self.dataset_path = dataset_path
        self.load_dataset()
        self.compute_overlap_runs()
        self.image_list = [i for j in self.overlap_runs for i in j]

    def load_dataset(self):

        def datetime(img):
            return img.datetime

        def blurry(pix_img):
            img = cv.imread(pix_img.img_path, 0)
            lap_variance = cv.Laplacian(img, cv.CV_64F).var()
            if lap_variance >= self.blur_threshold:
                return False
            return True

        self.__image_list = []
        images = os.listdir(self.dataset_path)

        for i in tqdm(range(len(images)), desc="Loading images"):
            pix_img = PixImage(os.path.join(self.dataset_path, images[i]))
            if blurry(pix_img) or pix_img.brightness_value < BRIGHTNESS_THRESHOLD:
                continue
            self.__image_list.append(pix_img)

        self.__image_list.sort(key=datetime)

    def compute_overlap_runs(self):

        self.overlap_runs = []
        tmp_list = []
        in_run = False

        cached_img = cv.imread(self.__image_list[0].img_path, 0)
        for i in tqdm(range(1, len(self.__image_list)-1), desc="Calculating overlaps"):
            img_1 = cached_img
            img_2 = cv.imread(self.__image_list[i].img_path, 0)
            cached_img = img_2

            try:
                overlap = calculate_overlap(img_1, img_2)
            except ValueError:
                if in_run:
                    tmp_list.append(self.__image_list[i-1])
                    self.overlap_runs.append(tmp_list)
                    tmp_list = []
                else:
                    continue

            if overlap >= OVERLAP_THRESHOLD:
                in_run = True
                if overlap >= 1:
                    continue
                tmp_list.append(self.__image_list[i-1])
            else:
                if in_run:
                    tmp_list.append(self.__image_list[i-1])
                    self.overlap_runs.append(tmp_list)
                    tmp_list = []
                    in_run = False
                else:
                    continue

            if i == len(self.__image_list)-1 and in_run:
                tmp_list.append(self.__image_list[i])
                self.overlap_runs.append(tmp_list)

    def save_dataset(self, output_dir):
        for i, p in enumerate(self.image_list):
            shutil.copyfile(p.img_path, f'{output_dir}/{i:03d}.jpg')

# TODO: Add iPhone image support
class PixImage:

    def __init__(self, img_path, gamma=GAMMA, blur_threshold=BLUR_THRESHOLD):

        self.img_path = img_path

        # Color/contrast correction parameters
        self.gamma = gamma
        self.blur_threshold = blur_threshold

        # Image metadata
        self.load_exif()

    def load_exif(self):
        try:
            with open(self.img_path, 'rb') as img_file:
                exif = Exif(self.img_path)
        except Exception:
            print("[ERROR]: No EXIF data exists")
            return

        self.W = exif.image_width
        self.H = exif.image_height
        self.pixels = self.W * self.H
        self.model = exif.model
        self.brightness_value = exif.brightness_value
        self.datetime = exif.datetime_original

    def increase_contrast(self):
        image = cv.imread(self.img_path, 0)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        lab_planes = cv.split(lab)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv.merge(lab_planes)
        contrast_img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

        return contrast_img

    def gamma_correct(self):
        image = cv.imread(self.img_path, 0)
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_img = cv.LUT(image, table)

        return gamma_img

if __name__ == '__main__':

    img_path = sys.argv[1]
    output_path = sys.argv[2]
    dataset = ImageDataset(img_path)
    dataset.save_dataset(output_path)
