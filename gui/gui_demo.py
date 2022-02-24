import sys
from loguru import logger
import filters
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QComboBox
from PySide6.QtCore import QObject,  Signal, Slot, Qt, QThread
from PySide6.QtGui import QImage, QPixmap
import inspect
import numpy as np
import cv2
import fps
from videoProcess import VideoLoopBack


class VideoThread(QObject):
    change_pixmap_signal = Signal(np.ndarray)
    update_fps_signal = Signal(float)

    def __init__(self,filter_list):
        super().__init__()
        self._fps = fps.FPS()
        self._run_flag = True
        self.filter_list = filter_list
        self.cap = cv2.VideoCapture(0)
        self._ind = 0
        self.video_loop = VideoLoopBack()


    @Slot()
    def new_filter_slot(self,ind):
        self._ind=ind
        logger.debug(f'ind {ind}')

    @Slot()
    def start(self):
        print('start')
        self._fps.start()
        while self._run_flag:
            ret, cv_img = self.cap.read()
            if ret:
                fimg = self.filter_list[self._ind][1].process(cv_img)
                self.video_loop.new_image(fimg)
                self.change_pixmap_signal.emit(fimg)
                self._fps.update()
                self.update_fps_signal.emit(self._fps.fps)

    @Slot()
    def pause(self):
        print('pause')
        self._run_flag=False


    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.cap.release()

#        self.wait()


class ControlWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setMinimumSize(400, 150)
        self.setFrameStyle(QFrame.Panel)



class CamImage(QLabel):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setMinimumSize(600,400)

class MainProcess():
    def __init__(self):
        self.list_of_filters= []
        self.blank_index=-1


    def get_image_functions(self):
        # filter_list = [o for o in getmembers(filters) if isc(o[1])]
        clsmembers = inspect.getmembers(filters, inspect.isclass)  # ,lambda x: isinstance(x, object))
        logger.info(f'Got: {clsmembers}')
        for c in clsmembers:
            logger.info(c[1])
            if issubclass(c[1], filters.BaseProcess) and not inspect.isabstract(c[1]):
                self.list_of_filters.append(c)
        logger.info(self.list_of_filters)
        tmp = [c[0] for c in self.list_of_filters]
        self.blank_index = tmp.index('Blank')



class App(QWidget):

    def __init__(self):
        super(App, self).__init__()
        self.combo= None
        self.MainProcess = MainProcess()
        self.MainProcess.get_image_functions()
        self.fps_label = QLabel('FPS: XXX')
        self.initUI()


        self._thread = QThread(self)
        self.video_worker = VideoThread(self.MainProcess.list_of_filters)
        #self.combo.currentIndexChanged.connect(self.video_worker.new_filter_slot)
        self.combo.currentIndexChanged.connect(self._new_ind)

        self.video_worker.change_pixmap_signal.connect(self.update_image)
        self.video_worker.update_fps_signal.connect(self._new_fps)
        self.video_worker.moveToThread(self._thread)
        self.combo.setCurrentIndex(self.MainProcess.blank_index)
        # connect the buttons
        self.start_button.clicked.connect(self.video_worker.start)
        self.pause_button.clicked.connect(lambda x: self.video_worker.pause()) # force this to run on current thread

        self._thread.start()

    @Slot()
    def _new_ind(self, ind):
        self.video_worker.new_filter_slot(ind)

    @Slot(int)
    def _new_fps(self, fps):
        self.fps_label.setText(f'FPS: {fps:.1f}')

    def initUI(self):

        self.cam = CamImage()
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Stop")
        controls = ControlWidget()
        self.combo = QComboBox(self)
        for it in self.MainProcess.list_of_filters:
            self.combo.addItem(it[0])


        hbox = QHBoxLayout()
        hbox.addWidget(controls)
        hbox.addStretch(1)
        hbuttons = QHBoxLayout()
        hbuttons.addWidget(self.combo)
        hbuttons.addWidget(self.start_button)
        hbuttons.addWidget(self.pause_button)
        vbutton = QVBoxLayout()
        vbutton.addLayout(hbuttons)
        vbutton.addWidget(self.fps_label)
        hbox.addLayout(vbutton)
        vbox = QVBoxLayout()
        vbox.addWidget(self.cam)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('Buttons')



        self.show()

    def closeEvent(self, event):
        self.video_worker.stop()
        event.accept()

    @Slot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.cam.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.cam.width(), self.cam.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
def main():

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()