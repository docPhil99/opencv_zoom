import ffmpeg
import subprocess
from loguru import logger
import numpy as np
from PySide6.QtCore import QObject,Signal, Slot
import cv2

class VideoLoopBack(QObject):
    def __init__(self,  out_filename='/dev/video5', width=640, height=480,
                 true_file=False, alt_filename=None):
        super().__init__()
        self.width = width
        self.height = height
        self.out_file = out_filename
        self.true_file = true_file
        logger.debug('Done init')
        self.alt_file = alt_filename
        self.__toggle_live = True
        self.process2 = self.start_ffmpeg_process_dev_video()

    def start_ffmpeg_process_dev_video(self):
        logger.info(f'Starting ffmpeg process2, writing to {self.out_file}')
        args = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(self.width, self.height))
                .output(self.out_file, pix_fmt='yuv420p', format='v4l2')
                .compile()
        )
        return subprocess.Popen(args, stdin=subprocess.PIPE)

    def write_frame(self,process2, frame):
        #logger.debug('Writing frame')
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        process2.stdin.write(
            frame2
                .astype(np.uint8)
                .tobytes()
        )

    @Slot(np.ndarray)
    def new_image(self, image):
        self.write_frame(self.process2,image)

