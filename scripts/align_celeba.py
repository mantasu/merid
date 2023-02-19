from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
from multiprocessing import Pool
import os
import re

import numpy as np
import tqdm

# ==============================================================================
# =                                      param                                 =
# ==============================================================================

parser = argparse.ArgumentParser()
# main
parser.add_argument('--img_dir', dest='img_dir', default='./data/celeba/img_celeba')
parser.add_argument('--save_dir', dest='save_dir', default='./data/celeba/aligned')
parser.add_argument('--landmark_file', dest='landmark_file', default='./data/celeba/landmark.txt')
parser.add_argument('--standard_landmark_file', dest='standard_landmark_file', default='./data/celeba/standard_landmark_68pts.txt')
parser.add_argument('--crop_size_h', dest='crop_size_h', type=int, default=572)
parser.add_argument('--crop_size_w', dest='crop_size_w', type=int, default=572)
parser.add_argument('--move_h', dest='move_h', type=float, default=0.25)
parser.add_argument('--move_w', dest='move_w', type=float, default=0.)
parser.add_argument('--save_format', dest='save_format', choices=['jpg', 'png'], default='jpg')
parser.add_argument('--n_worker', dest='n_worker', type=int, default=8)
# others
parser.add_argument('--face_factor', dest='face_factor', type=float, help='The factor of face area relative to the output image.', default=0.45)
parser.add_argument('--align_type', dest='align_type', choices=['affine', 'similarity'], default='similarity')
parser.add_argument('--order', dest='order', type=int, choices=[0, 1, 2, 3, 4, 5], help='The order of interpolation.', default=3)
parser.add_argument('--mode', dest='mode', choices=['constant', 'edge', 'symmetric', 'reflect', 'wrap'], default='edge')
args = parser.parse_args()

# ==============================================================================
# =                                   cropper                                  =
# ==============================================================================

def align_crop_opencv(img,
                      src_landmarks,
                      standard_landmarks,
                      crop_size=512,
                      face_factor=0.7,
                      align_type='similarity',
                      order=3,
                      mode='edge'):
    """Align and crop a face image by landmarks.

    Arguments:
        img                : Face image to be aligned and cropped.
        src_landmarks      : [[x_1, y_1], ..., [x_n, y_n]].
        standard_landmarks : Standard shape, should be normalized.
        crop_size          : Output image size, should be 1. int for (crop_size, crop_size)
                             or 2. (int, int) for (crop_size_h, crop_size_w).
        face_factor        : The factor of face area relative to the output image.
        align_type         : 'similarity' or 'affine'.
        order              : The order of interpolation. The order has to be in the range 0-5:
                                 - 0: INTER_NEAREST
                                 - 1: INTER_LINEAR
                                 - 2: INTER_AREA
                                 - 3: INTER_CUBIC
                                 - 4: INTER_LANCZOS4
                                 - 5: INTER_LANCZOS4
        mode               : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                             Points outside the boundaries of the input are filled according
                             to the given mode.
    """
    # set OpenCV
    inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
             3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
    border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
              'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
              'wrap': cv2.BORDER_WRAP}

    # check
    assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant', 'edge', 'symmetric', 'reflect', 'wrap']

    # crop size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        crop_size_h = crop_size[0]
        crop_size_w = crop_size[1]
    elif isinstance(crop_size, int):
        crop_size_h = crop_size_w = crop_size
    else:
        raise Exception('Invalid `crop_size`! `crop_size` should be 1. int for (crop_size, crop_size) or 2. (int, int) for (crop_size_h, crop_size_w)!')

    # estimate transform matrix
    trg_landmarks = standard_landmarks * max(crop_size_h, crop_size_w) * face_factor + np.array([crop_size_w // 2, crop_size_h // 2])
    if align_type == 'affine':
        tform = cv2.estimateAffine2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]
    else:
        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]

    # warp image by given transform
    output_shape = (crop_size_h, crop_size_w)
    img_crop = cv2.warpAffine(img, tform, output_shape[::-1], flags=cv2.WARP_INVERSE_MAP + inter[order], borderMode=border[mode])

    # get transformed landmarks
    tformed_landmarks = cv2.transform(np.expand_dims(src_landmarks, axis=0), cv2.invertAffineTransform(tform))[0]

    return img_crop, tformed_landmarks


def align_crop_skimage(img,
                       src_landmarks,
                       standard_landmarks,
                       crop_size=512,
                       face_factor=0.7,
                       align_type='similarity',
                       order=3,
                       mode='edge'):
    """Align and crop a face image by landmarks.

    Arguments:
        img                : Face image to be aligned and cropped.
        src_landmarks      : [[x_1, y_1], ..., [x_n, y_n]].
        standard_landmarks : Standard shape, should be normalized.
        crop_size          : Output image size, should be 1. int for (crop_size, crop_size)
                             or 2. (int, int) for (crop_size_h, crop_size_w).
        face_factor        : The factor of face area relative to the output image.
        align_type         : 'similarity' or 'affine'.
        order              : The order of interpolation. The order has to be in the range 0-5:
                                 - 0: INTER_NEAREST
                                 - 1: INTER_LINEAR
                                 - 2: INTER_AREA
                                 - 3: INTER_CUBIC
                                 - 4: INTER_LANCZOS4
                                 - 5: INTER_LANCZOS4
        mode               : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                             Points outside the boundaries of the input are filled according
                             to the given mode.
    """
    raise NotImplementedError("'align_crop_skimage' is not implemented!")

# ==============================================================================
# =                                opencv first                                =
# ==============================================================================

_DEAFAULT_JPG_QUALITY = 95
try:
    import cv2
    imread = cv2.imread
    imwrite = partial(cv2.imwrite, params=[int(cv2.IMWRITE_JPEG_QUALITY), _DEAFAULT_JPG_QUALITY])
    align_crop = align_crop_opencv
    print('Use OpenCV')
except:
    import skimage.io as io
    imread = io.imread
    imwrite = partial(io.imsave, quality=_DEAFAULT_JPG_QUALITY)
    align_crop = align_crop_skimage
    print('Importing OpenCv fails. Use scikit-image')

# ==============================================================================
# =                                     run                                    =
# ==============================================================================

# count landmarks
with open(args.landmark_file) as f:
    line = f.readline()
n_landmark = len(re.split('[ ]+', line)[1:]) // 2

# read data
img_names = np.genfromtxt(args.landmark_file, dtype=str, usecols=0)
landmarks = np.genfromtxt(args.landmark_file, dtype=float, usecols=range(1, n_landmark * 2 + 1)).reshape(-1, n_landmark, 2)
standard_landmark = np.genfromtxt(args.standard_landmark_file, dtype=float).reshape(n_landmark, 2)
standard_landmark[:, 0] += args.move_w
standard_landmark[:, 1] += args.move_h

# data dir
save_dir = os.path.join(args.save_dir, 'align_size(%d,%d)_move(%.3f,%.3f)_face_factor(%.3f)_%s' % (args.crop_size_h, args.crop_size_w, args.move_h, args.move_w, args.face_factor, args.save_format))
data_dir = os.path.join(save_dir, 'data')
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)


def work(i):  # a single work
    for _ in range(3):  # try three times
        try:
            img = imread(os.path.join(args.img_dir, img_names[i]))
            img_crop, tformed_landmarks = align_crop(img,
                                                     landmarks[i],
                                                     standard_landmark,
                                                     crop_size=(args.crop_size_h, args.crop_size_w),
                                                     face_factor=args.face_factor,
                                                     align_type=args.align_type,
                                                     order=args.order,
                                                     mode=args.mode)

            name = os.path.splitext(img_names[i])[0] + '.' + args.save_format
            path = os.path.join(data_dir, name)
            if not os.path.isdir(os.path.split(path)[0]):
                os.makedirs(os.path.split(path)[0])
            imwrite(path, img_crop)

            tformed_landmarks.shape = -1
            name_landmark_str = ('%s' + ' %.1f' * n_landmark * 2) % ((name, ) + tuple(tformed_landmarks))
            succeed = True
            break
        except:
            succeed = False
    if succeed:
        return name_landmark_str
    else:
        print('%s fails!' % img_names[i])


pool = Pool(args.n_worker)
name_landmark_strs = list(tqdm.tqdm(pool.imap(work, range(len(img_names))), total=len(img_names)))
pool.close()
pool.join()

landmarks_path = os.path.join(save_dir, 'landmark.txt')
with open(landmarks_path, 'w') as f:
    for name_landmark_str in name_landmark_strs:
        if name_landmark_str:
            f.write(name_landmark_str + '\n')
