import datetime
from loguru import logger


class FPS:
    """
    Simple class for calculating and printing the frames per second
    """

    def __init__(self, frames_to_update=10, verbose=False, name=None):
        """

        :param frames_to_update: how many frames to use to calculate the fps, fps is undefined before the first
        calculation
        :param verbose: set to True if you want it to print to the logger info stream
        :param name: set a string if you want to include a name if verbose is set
        """
        self.counter=0
        self._start_time = None
        self._frames_to_update = frames_to_update
        self._fps = -1
        self._verbose = verbose
        self._name = name
        if not self._name:
            self._name = "Unknown"

    def start(self):
        """
        Call before the first loop
        :return:
        """
        self._start_time = datetime.datetime.now()

    def update(self):
        """
        call every frame
        :return:
        """
        if not self._start_time:
            self.start()

        self.counter += 1

        if self.counter % self._frames_to_update == 0:
            time = ( datetime.datetime.now()- self._start_time ).total_seconds()
            self._fps = self.counter/time
            if self._verbose:
                logger.info(f'FPS from {self._name} {self._fps}')
            self.counter = 0
            self.start()

    @property
    def fps(self):
        """
        the frames per second
        """
        return self._fps


if __name__ == "__main__":
    """test"""
    import time
    fp=FPS()
    fp.start()

    for p in range(30):
        fp.update()
        print(f'Fps {fp.fps}')
        time.sleep(0.5)
