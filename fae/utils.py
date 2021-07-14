import cv2
import gdown
import os.path
import torch
import warnings
import numpy as np
import string
from .decode import CtcDecoder
from jiwer import wer
import face_alignment
import torch.nn.functional as F
from skimage import transform as tf


def read_video(filename, size=None):
    cap = cv2.VideoCapture(filename)
    while (cap.isOpened()):
        ret, frame = cap.read()  # BGR
        if ret:
            if size is not None and (size[0] != frame.shape[0] or size[1] != frame.shape[1]):
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            break
    cap.release()


class MouthEvaluator:
    def __init__(self, lipreader=None, device="cuda", label_map=None):

        alphabet = ['_'] + list(string.ascii_uppercase) + [' ']
        self.ctc_decoder = CtcDecoder(alphabet)
        self.stable_pt_ids = [33, 36, 39, 42, 45]
        self.mouth_pt_ids = range(48, 68)
        self.size = (88, 88)
        self.device = torch.device(device)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)
        self.mean_face = np.load(os.path.split(__file__)[0] + "/resources/mean_face.npy")

        if lipreader == "lrw":
            if not os.path.exists(os.path.split(__file__)[0] + "/resources/lrw.pth"):
                url = 'https://drive.google.com/uc?id=1oYDAhvYyuFydkECuPJlaDMAl5n_v3jTj'
                output = os.path.split(__file__)[0] + "/resources/lrw.pth"
                gdown.download(url, output, quiet=False)

            lipreader = os.path.split(__file__)[0] + "/resources/lrw.pth"
            label_map = os.path.split(__file__)[0] + "/resources/500WordsSortedList.txt"

        self.label_map = None
        if label_map is not None:
            self.label_map = {}
            with open(label_map) as f:
                annotation_list = f.readlines()
                for l, a in enumerate(annotation_list):
                    self.label_map[l] = a.rstrip()

        if lipreader is None:
            self.lipreader = None
        else:
            self.lipreader = torch.jit.load(lipreader).to(self.device)
            self.lipreader.eval()

    def warp(self, src, dst, img, std_size=(256, 256)):
        tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
        warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped, tform

    def cut_patch(self, img, landmarks, height, width, threshold=5):

        center_x, center_y = np.mean(landmarks, axis=0)

        if center_y - height < 0:
            center_y = height
        if center_y - height < 0 - threshold:
            raise Exception('too much bias in height')
        if center_x - width < 0:
            center_x = width
        if center_x - width < 0 - threshold:
            raise Exception('too much bias in width')

        if center_y + height > img.shape[0]:
            center_y = img.shape[0] - height
        if center_y + height > img.shape[0] + threshold:
            raise Exception('too much bias in height')
        if center_x + width > img.shape[1]:
            center_x = img.shape[1] - width
        if center_x + width > img.shape[1] + threshold:
            raise Exception('too much bias in width')

        cut_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                          int(round(center_x) - round(width)): int(round(center_x) + round(width))])
        return cut_img

    def __call__(self, video_path, ref_video_path=None, annotation=None):
        if (self.lipreader is None or annotation is None) and ref_video_path is None:
            warnings.simplefilter("once")
            warnings.warn("You have neither provided a lipreader nor a reference video so there will be no mouth movement evaluation")
            return {}

        metrics = {}
        video = read_video(video_path)
        if ref_video_path is not None:
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            ref_video = read_video(ref_video_path, (height, width))
            lmd = 0
            ref_sequence = []

        sequence = []
        no_frames = 0
        for frame in video:
            no_frames += 1
            try:
                frame_landmarks = self.fa.get_landmarks(frame)[0]
                if frame_landmarks is None:
                    return {}
            except:
                return {}

            if self.lipreader is not None:
                trans_frame, trans = self.warp(frame_landmarks[self.stable_pt_ids, :], self.mean_face[self.stable_pt_ids, :], frame)
                trans_landmarks = trans(frame_landmarks)

                gray_frame = cv2.cvtColor(self.cut_patch(trans_frame, trans_landmarks[self.mouth_pt_ids], 48, 48), cv2.COLOR_RGB2GRAY)
                norm_frame = (gray_frame / 255 - 0.421) / 0.165
                h, w = norm_frame.shape
                th, tw = self.size
                delta_w = int(round((w - tw)) / 2.)
                delta_h = int(round((h - th)) / 2.)
                cropped_norm_frame = norm_frame[delta_h:delta_h + th, delta_w:delta_w + tw]

                sequence.append(torch.Tensor(cropped_norm_frame).unsqueeze(0))

            if ref_video_path is not None:
                ref_frame = next(ref_video)
                ref_frame_landmarks = self.fa.get_landmarks(ref_frame)[0]

                if self.lipreader is not None:
                    trans_frame, trans = self.warp(ref_frame_landmarks[self.stable_pt_ids, :], self.mean_face[self.stable_pt_ids, :], ref_frame)
                    trans_landmarks = trans(ref_frame_landmarks)
                    gray_frame = cv2.cvtColor(self.cut_patch(trans_frame, trans_landmarks[self.mouth_pt_ids], 48, 48), cv2.COLOR_RGB2GRAY)
                    norm_frame = (gray_frame / 255 - 0.421) / 0.165
                    h, w = norm_frame.shape
                    th, tw = self.size
                    delta_w = int(round((w - tw)) / 2.)
                    delta_h = int(round((h - th)) / 2.)
                    cropped_norm_frame = norm_frame[delta_h:delta_h + th, delta_w:delta_w + tw]

                    ref_sequence.append(torch.Tensor(cropped_norm_frame).unsqueeze(0))

                lmd += ref_frame_landmarks[self.mouth_pt_ids] - frame_landmarks[self.mouth_pt_ids].mean()

        lip_video = torch.cat(sequence)
        if ref_video_path is not None:
            ref_lip_video = torch.cat(ref_sequence)
            metrics["LMD"] = lmd / no_frames

        if self.lipreader is not None:
            with torch.no_grad():
                logits = self.lipreader(lip_video.unsqueeze(0).unsqueeze(0).to(self.device), torch.LongTensor([no_frames]))
                _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
                # With sentence level lip_reading use the following code
                # sentence = self.ctc_decoder.decode_batch(probs.detach().cpu(), torch.LongTensor([probs.size(0)]))[0]
                predicted = predicted.detach().cpu().numpy()[0]
                if self.label_map is not None:
                    predicted = self.label_map[predicted]

            if annotation is None:
                warnings.warn("Annotations not provided using reference video prediction as label")
                ref_logits = self.lipreader(ref_lip_video.unsqueeze(0).unsqueeze(0).to(self.device), torch.LongTensor([no_frames]))
                _, gt = torch.max(F.softmax(ref_logits, dim=1).data, dim=1)
                if self.label_map is not None:
                    gt = self.label_map[gt]
            else:
                gt = annotation

            metrics["WER"] = wer(gt.lower(), predicted.lower())

        return metrics
