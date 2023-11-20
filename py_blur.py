
%%file py_face_blur_module.py

import bmf
import numpy as np
from bmf import ProcessResult, Packet, Timestamp, VideoFrame
import PIL
from PIL import Image
import bmf.hml.hmp as mp
import cv2
import warnings

debug = False

class py_face_blur_module(bmf.Module):
    def __init__(self, node, option=None):
        print(f'py_face_blur_module init ...')
        self.node_ = node
        self.option_ = option
        print(option)
        warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.idx = 0

        print(f'py_face_blur_module init successfully...')


    def process(self, task):
        idx = self.idx

        for (input_id, input_queue) in task.get_inputs().items():
            output_queue = task.get_outputs()[input_id]

            while not input_queue.empty():
                packet = input_queue.get()

                if packet.timestamp == Timestamp.EOF:
                    output_queue.put(Packet.generate_eof_packet())
                    task.timestamp = Timestamp.DONE

                if packet.timestamp != Timestamp.UNSET and packet.is_(VideoFrame):

                    vf = packet.get(VideoFrame)
                    rgb = mp.PixelInfo(mp.kPF_RGB24)
                    np_vf = vf.reformat(rgb).frame().plane(0).numpy()

                    # numpy to PIL
                    image = Image.fromarray(np_vf.astype('uint8'), 'RGB')

                    # PIL to OpenCV
                    open_cv_image = np.array(image)
                    open_cv_image = open_cv_image[:, :, ::-1].copy()

                    # Detect faces
                    faces = self.face_cascade.detectMultiScale(open_cv_image, 1.1, 4)

                    # Blur faces
                    for (x, y, w, h) in faces:
                        face = open_cv_image[y:y+h, x:x+w]
                        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
                        open_cv_image[y:y+h, x:x+w] = blurred_face

                    # OpenCV to PIL
                    blurred_image = Image.fromarray(open_cv_image[:, :, ::-1])

                    if debug:
                        input_name = f'video/bmf_raw/frame_{idx}.png'
                        print(f'input_name = {input_name}')
                        image.save(input_name)

                        output_name = f'video/bmf_out/frame_{idx}.png'
                        print(f'output_name = {output_name}')
                        blurred_image.save(output_name)

                    self.idx = idx + 1
                    out_frame_np = np.array(blurred_image)
                    rgb = mp.PixelInfo(mp.kPF_RGB24)
                    frame = mp.Frame(mp.from_numpy(out_frame_np), rgb)

                    out_frame = VideoFrame(frame)
                    out_frame.pts = vf.pts
                    out_frame.time_base = vf.time_base

                    pkt = Packet(out_frame)
                    pkt.timestamp = out_frame.pts

                    output_queue.put(pkt)

        return ProcessResult.OK


