import argparse
import warnings

import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from tqdm import tqdm

from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class GanInference:
    def __init__(self, input_folder='inputs/whole_imgs', output_folder='results', version='1.3', upscale=2,
                 bg_upsampler='realesrgan', bg_tile=400, suffix=None, only_center_face=False, aligned=False,
                 save_faces=False, extension='auto'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.version = version
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler
        self.bg_tile = bg_tile
        self.suffix = suffix
        self.only_center_face = only_center_face
        self.aligned = aligned
        self.save_faces = save_faces
        self.extension = extension
        self.processedFramesFolderPath = os.path.join('temp', 'processedFrames')
        self.unprocessedFramesFolderPath = os.path.join('temp', 'unprocessedFrames')
        self.restoredFramesFolderPath = os.path.join('temp', 'restoredFrames')

    def load_video(self, video_path):
        vidcap = cv2.VideoCapture(video_path)

        # Get the total number of frames in the video
        numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the frames per second (fps) of the video
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("FPS: ", fps, "Frames: ", numberOfFrames)  # Print the fps and the total number of frames
        from tqdm import tqdm
        # Loop over each frame in the video
        for frameNumber in tqdm(range(numberOfFrames)):
            # Read the next frame from the video
            _, image = vidcap.read()
            # Save the frame as a .jpg image in the specified folder
            cv2.imwrite(os.path.join(self.unprocessedFramesFolderPath, str(frameNumber).zfill(4) + '.jpg'), image)
            if numberOfFrames == 10:
                break

    def run_inference(self):
        if self.input_folder.endswith('/'):
            self.input_folder = self.input_folder[:-1]
        if os.path.isfile(self.input_folder):
            img_list = [self.input_folder]
        else:
            img_list = sorted(glob.glob(os.path.join(self.input_folder, '*')))

        os.makedirs(self.output_folder, exist_ok=True)

        if self.bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                              'If you really want to use it, please modify the corresponding codes.')
                bg_upsampler = None
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=self.bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)
        else:
            bg_upsampler = None

        if self.version == '1':
            arch = 'original'
            channel_multiplier = 1
            model_name = 'GFPGANv1'
        elif self.version == '1.2':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANCleanv1-NoCE-C2'
        elif self.version == '1.3':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.3'
        else:
            raise ValueError(f'Wrong model version {self.version}.')

        model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('realesrgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            raise ValueError(f'Model {model_name} does not exist.')

        restorer = GFPGANer(
            model_path=model_path,
            upscale=self.upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)

        for img_path in tqdm(img_list):
            img_name = os.path.basename(img_path)
            print(f'Processing {img_name} ...')
            basename, ext = os.path.splitext(img_name)
            input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            cropped_faces, restored_faces, restored_img = restorer.enhance(
                input_img, has_aligned=self.aligned, only_center_face=self.only_center_face, paste_back=True)

            if self.save_faces:
                for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                    save_crop_path = os.path.join(self.output_folder, 'cropped_faces', f'{basename}_{idx:02d}.png')
                    imwrite(cropped_face, save_crop_path)
                    if self.suffix is not None:
                        save_face_name = f'{basename}_{idx:02d}_{self.suffix}.png'
                    else:
                        save_face_name = f'{basename}_{idx:02d}.png'
                    save_restore_path = os.path.join(self.output_folder, 'restored_faces', save_face_name)
                    imwrite(restored_face, save_restore_path)
                    cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                    imwrite(cmp_img, os.path.join(self.output_folder, 'cmp', f'{basename}_{idx:02d}.png'))

            if restored_img is not None:
                if self.extension == 'auto':
                    extension = ext[1:]
                else:
                    extension = self.extension

                if self.suffix is not None:
                    save_restore_path = os.path.join(self.output_folder, 'restored_imgs',
                                                     f'{basename}_{self.suffix}.{extension}')
                else:
                    save_restore_path = os.path.join(self.output_folder, 'restored_imgs',
                                                     f'{basename}.{extension}')
                imwrite(restored_img, save_restore_path)

        print(f'Results are in the [{self.output_folder}] folder.')

# Example usage
gan_inference = GanInference(input_folder='inputs/whole_imgs', output_folder='results', version='1.3', upscale=2,
                             bg_upsampler='realesrgan', bg_tile=400, suffix=None, only_center_face=False,
                             aligned=False, save_faces=False, extension='auto')
gan_inference.load_video('results/test123.mp4')

