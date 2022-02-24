'''
Example streaming ffmpeg numpy processing.
Demonstrates using ffmpeg to decode video input, process the frames in
python/opencv, and then encode the video output to /dev/video.

run  sudo apt install v4l2loopback-dkms  

Before starting run sudo modprobe v4l2loopback \
      devices=1 exclusive_caps=1 video_nr=5 \
      card_label="Dummy Camera"

      Start this running before Zoom so zoom detects the camera. You can then start and stop again as you need.

'''

import cv2
import ffmpeg
import logging
import numpy as np
import subprocess
from inspect import getmembers, isfunction
from filters import *
import filters
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class VideoLoopBack:
    def __init__(self, in_filename='/dev/video0',out_filename='/dev/video5', width=640,height=480,process_fnc=colormap,
                 true_file=False, alt_filename=None):
        self.width = width
        self.height = height
        self.process_fcn = process_fnc
        self.in_file = in_filename
        self.out_file = out_filename
        self.true_file = true_file
        logger.debug('Done init')
        self.alt_file = alt_filename
        self.__toggle_live = True


    def start_ffmpeg_alt_file(self):
        logger.info(f'Starting ffmpeg process1 from {self.alt_file}')

        args = (
            ffmpeg
                .input(self.alt_file, re=None, stream_loop='-1')
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', vf=f'scale={self.width}x{self.height}')
                .compile()
        )
        return subprocess.Popen(args, stdout=subprocess.PIPE)

    def start_ffmpeg_webcam(self):
        logger.info(f'Starting ffmpeg process1 from {self.in_file}')
        if self.true_file:

            args = (
                ffmpeg
                    #.input(self.in_file,re=None,stream_loop='-1',fflags='nobuffer',flags='low_delay')
                    .input(self.in_file, framerate=15, stream_loop='-1', fflags='nobuffer', flags='low_delay')

                    .output('pipe:', format='rawvideo', pix_fmt='rgb24',vf=f'scale={self.width}x{self.height}')
                    .compile()
            )
        else:
            args = (
                ffmpeg
                    .input(self.in_file, f='v4l2',framerate= '25', video_size='{}x{}'.format(self.width, self.height), fflags='nobuffer',flags='low_delay')
                    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                    .compile()
            )
        logger.debug(f'FFmpeg input {args}')
        return subprocess.Popen(args, stdout=subprocess.PIPE)



    def start_ffmpeg_process_dev_video(self):
        logger.info(f'Starting ffmpeg process2, writing to {self.out_file}')
        args = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(self.width, self.height))
                .output(self.out_file, pix_fmt='yuv420p',format='v4l2')
                .compile()
        )
        return subprocess.Popen(args, stdin=subprocess.PIPE)



    def read_frame(self,process1):
        logger.debug('Reading frame')
        # Note: RGB24 == 3 bytes per pixel.
        frame_size = self.width * self.height * 3
        in_bytes = process1.stdout.read(frame_size)
        if len(in_bytes) == 0:
            frame = None
        else:
            assert len(in_bytes) == frame_size
            frame = (
                np
                    .frombuffer(in_bytes, np.uint8)
                    .reshape([self.height, self.width, 3])
            )
        process1.stdout.flush()
        return frame

    def write_frame(self,process2, frame):
        logger.debug('Writing frame')

        process2.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )


    def run(self):
        process1 = self.start_ffmpeg_webcam( )
        process2 = self.start_ffmpeg_process_dev_video()
        if self.alt_file:
            process_alt = self.start_ffmpeg_alt_file()
        else:
            process_alt = None

        filter_list = [o for o in getmembers(filters) if isfunction(o[1])]

        while True:

            # read live feed
            if self.__toggle_live:
                in_frame = self.read_frame(process1)
                if in_frame is None:
                    logger.info('End of input stream')
                    break
            else:
                # read file feed
                if process_alt:
                    alt_frame = self.read_frame(process_alt)
                    if alt_frame is None:
                        logger.info('End of alt input stream')
                        break

            logger.debug('Processing frame')
            if self.__toggle_live:
                out_frame = self.process_fcn(in_frame)
            else:
                out_frame = self.process_fcn(alt_frame)

            self.write_frame(process2, out_frame)
            # handle local display
            dirty_frame = np.copy(out_frame) # deep copy
            dirty_frame = cv2.cvtColor(dirty_frame,cv2.COLOR_BGR2RGB)
            for idx,f in enumerate(filter_list):
                cv2.putText(dirty_frame,f"{idx}:  {f[0]}",(10,30*idx+30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),1)
            cv2.imshow("test", dirty_frame)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            if key == ord('l') and process_alt:
                self.__toggle_live = not self.__toggle_live

            key -= 48
            if key>=0 and key < len(filter_list):
                print(f"Running process {filter_list[key][0]} id {filter_list[key][1]}")
                self.process_fcn = filter_list[key][1]

           # print(key)

        process2.stdin.close()
        logger.info('Waiting for ffmpeg process1')
        process1.wait()

        logger.info('Waiting for ffmpeg process2')

        process2.wait()
        logger.info('Done')


if __name__ == '__main__':
    #import curses

    #stdscr = curses.initscr()
    #stdscr.nodelay(True)

    logger.info('Started')
    logger.info([o for o in getmembers(filters) if isfunction(o[1])])
    VideoLoopBack(process_fnc=none, alt_filename='videos/monty1.gif', true_file=True).run()
    #VideoLoopBack(process_fnc=none, in_filename='/home/phil/Videos/Speed1.avi',true_file=True).run()
    #VideoLoopBack(process_fnc=temporal_feedback).run()
