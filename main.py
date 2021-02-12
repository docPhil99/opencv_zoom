# Simple play video file example.
import cv2
import ffmpeg
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def process_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.Sobel(img, cv2.CV_16S, 0, 1, 3, scale=1)
    img = cv2.convertScaleAbs(img)
    cv2.putText(img, "sobely", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    cv2.imshow("Test", img)

def from_file():
    in_filename='/home/phil/Videos/Speed1.avi'
    probe = ffmpeg.probe(in_filename)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    num_frames = int(video_stream['nb_frames'])

    out, _ = (
        ffmpeg
            .input(in_filename)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True)
    )
    video = (
        np
            .frombuffer(out, np.uint8)
            .reshape([-1, height, width, 3])

    )

    
def test_cap():
    cap = cv2.VideoCapture(0)

    myFuncs = [process_frame]
    ind = 0
    while (True):
        ret, frame = cap.read()

        myFuncs[ind](frame)
        cv2.imshow("Live", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n"):
            ind += 1
            ind %= len(myFuncs)

    cap.release()
    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    test_cap()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

