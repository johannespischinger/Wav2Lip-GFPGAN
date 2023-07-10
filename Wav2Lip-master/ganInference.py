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
    def __init__(self, videoPath, audioPath,version='1.3', upscale=1,
                 bg_upsampler='realesrgan', bg_tile=400, suffix=None, only_center_face=False, aligned=False, extension='auto'):

        self.version = version
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler
        self.bg_tile = bg_tile
        self.suffix = suffix
        self.only_center_face = only_center_face
        self.aligned = aligned
        self.extension = extension
        self.processedFramesFolderPath = os.path.join('temp', 'processedFrames')
        self.unprocessedFramesFolderPath = os.path.join('temp', 'unprocessedFrames')
        self.restoredFramesFolderPath = os.path.join(self.processedFramesFolderPath, 'restoredFrames')
        self.videoPath = videoPath
        self.audioPath = audioPath


    def load_video(self):
        vidcap = cv2.VideoCapture(self.videoPath)

        # Get the total number of frames in the video
        numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the frames per second (fps) of the video
        self.fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("FPS: ", self.fps, "Frames: ", numberOfFrames)  # Print the fps and the total number of frames
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
        if self.unprocessedFramesFolderPath.endswith('/'):
            self.unprocessedFramesFolderPath = self.unprocessedFramesFolderPath[:-1]
        if os.path.isfile(self.unprocessedFramesFolderPath):
            img_list = [self.unprocessedFramesFolderPath]
        else:
            img_list = sorted(glob.glob(os.path.join(self.unprocessedFramesFolderPath, '*')))

        os.makedirs(self.processedFramesFolderPath, exist_ok=True)

        if self.bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                              'If you really want to use it, please modify the corresponding codes.')


            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='checkpoints/RealESRGAN_x2plus.pth',
                model=model,
                tile=self.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=False)
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

        model_path = os.path.join('checkpoints', model_name + '.pth')
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

            if restored_img is not None:
                if self.extension == 'auto':
                    extension = ext[1:]
                else:
                    extension = self.extension

                if self.suffix is not None:
                    save_restore_path = os.path.join(self.restoredFramesFolderPath,
                                                     f'{basename}_{self.suffix}.{extension}')
                else:
                    save_restore_path = os.path.join(self.restoredFramesFolderPath,
                                                     f'{basename}.{extension}')
                imwrite(restored_img, save_restore_path)

        print(f'Results are in the [{self.processedFramesFolderPath}] folder.')

    def createFinalVideo(self):
        tempPath = 'temp'
        if not os.path.exists(tempPath):
            os.makedirs(tempPath)

        dir_list = os.listdir(self.restoredFramesFolderPath)
        dir_list.sort()

        batch = 0
        batchSize = 300

        for i in tqdm(range(0, len(dir_list), batchSize)):
            img_array = []
            start, end = i, i + batchSize
            print("processing ", start, end)
            for filename in tqdm(dir_list[start:end]):
                filename = os.path.join(self.restoredFramesFolderPath, filename)
                img = cv2.imread(filename)
                if img is None:
                    continue
                height, width, layers = img.shape
                size = (width, height)
                img_array.append(img)

            out = cv2.VideoWriter(tempPath + '/batch_' + str(batch).zfill(4) + '.mp4', cv2.VideoWriter_fourcc(*'DIVX'), self.fps, size)
            batch = batch + 1

            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()

        ################################

        concatTextFilePath = os.path.join(tempPath, "concat.txt")  # Path to the text file for concatenation
        # Check if the folder to save the frames exists, if not, create it

        concatTextFile = open(concatTextFilePath, "w")  # Open the file in write mode

        # Write file paths of individual video files to be concatenated
        for ips in range(batch):
            concatTextFile.write("file batch_" + str(ips).zfill(4) + ".mp4\n")

        concatTextFile.close()  # Close the text file

        concatedVideoOutputPath = os.path.join(tempPath, 'concated_output.mp4')  # Output path for the concatenated video

        # Concatenate the videos using FFmpeg command
        import subprocess
        command = ['ffmpeg', '-y', '-f', 'concat', '-i', 'temp/concat.txt', '-c', 'copy', concatedVideoOutputPath]
        subprocess.run(command)


        # Run ffprobe command to get video duration
        ffprobe_cmd = f'ffprobe -i "{self.videoPath}" -show_entries format=duration -v quiet -of csv="p=0"'
        duration_bytes = subprocess.check_output(ffprobe_cmd, shell=True)
        duration = float(duration_bytes.decode("utf-8").strip())

        # Use the duration in the ffmpeg command to concatenate the concatedVideoOutputPath with input audio
        ffmpeg_cmd = f'ffmpeg -y -i "{concatedVideoOutputPath}" -i {self.audioPath} -map 0 -map 1:a -c:v copy -t {duration} {self.videoPath}'
        subprocess.call(ffmpeg_cmd, shell=True)

    def run(self):
        self.load_video()
        self.run_inference()
        self.createFinalVideo()

# # Example usage
# gan_inference = GanInference(videoPath='inputs/test.mp4', audioPath= 'inputs/CID016_Jakob.wav',version='1.3', upscale=1,
#                              bg_upsampler='realesrgan', bg_tile=400, suffix=None, only_center_face=False,
#                              aligned=False, extension='auto')
# gan_inference.load_video('results/test123.mp4')
# gan_inference.run_inference()
# gan_inference.createFinalVideo()

