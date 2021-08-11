import cv2
import gdown
import os.path
import torch
import warnings
import numpy as np
import string
import itertools
from bidict import bidict
from collections import Counter
from .decode import CtcDecoder
from jiwer import wer
import torch.nn.functional as F
from skimage import transform as tf
import face_recognition
from torchvision import transforms


def read_video(video):
    if isinstance(video, str):
        cap = cv2.VideoCapture(video)
        while (cap.isOpened()):
            ret, frame = cap.read()  # BGR
            if ret:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                break
        cap.release()
    else:
        vid = np.rollaxis((video * 0.5 + 0.5) * 255, 1, 4).astype(np.uint8)
        for frame in vid:
            yield frame


class EmotionEvaluator:
    def __init__(self, emotion_recognizer="emonet", num_emotions=8, device="cpu", label_map=None, resize=None, crop=False, ignore_emotions=None,
                 aggregation="voting"):
        self.device = torch.device(device)
        if emotion_recognizer == "emonet":
            if num_emotions == 5:
                model_path = os.path.split(__file__)[0] + "/resources/emotions5.pth"
                if not os.path.exists(model_path):
                    url = 'https://drive.google.com/uc?id=1_9HbsktpPveMxDT26d-Ns1VF6RAjHibc'
                    gdown.download(url, model_path, quiet=False)
                label_map = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'anger'}
            else:
                model_path = os.path.split(__file__)[0] + "/resources/emotions8.pth"

                if not os.path.exists(model_path):
                    url = 'https://drive.google.com/uc?id=1eLXyyuStCwNVNwaE5JkoJ-dR55lADSVH'
                    gdown.download(url, model_path, quiet=False)
                label_map = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt', 8: 'none'}

            resize = (256, 256)
            crop = True
            emotion_recognizer = model_path

        self.aggregation = aggregation
        self.transform = transforms.ToTensor()
        self.label_map = bidict(label_map)
        self.crop = crop
        self.resize = resize
        self.ignore_idxs = []
        if ignore_emotions is not None:
            for emotion in ignore_emotions:
                self.ignore_idxs.append(self.label_map.inv[emotion])

        self.emotion_recogniser = torch.jit.load(emotion_recognizer).to(self.device)
        self.emotion_recogniser.eval()

    def pcc(self, gt, pred):
        return np.corrcoef(gt, pred)[0, 1]

    def ccc(self, gt, pred):
        mean_pred = np.mean(pred)
        mean_gt = np.mean(gt)

        std_pred = np.std(pred)
        std_gt = np.std(gt)

        pcc = self.pcc(gt, pred)
        return 2.0 * pcc * std_pred * std_gt / (std_pred ** 2 + std_gt ** 2 + (mean_pred - mean_gt) ** 2)

    def to(self, device):
        self.device = device
        self.emotion_recogniser.to(device)

    def __call__(self, vid, ref_vid=None, annotation=None, crop=True, valence=None, arousal=None):
        video = read_video(vid)
        metrics = {}

        with torch.no_grad():
            votes = {}
            avg_logits = 0
            frame_number = 0
            val_sequence = []
            ar_sequence = []
            for frame in video:
                if self.crop:
                    face_loc = face_recognition.face_locations(frame)
                    if face_loc:
                        loc = face_loc[0]
                    else:
                        loc = (0, -1, -1, 0)
                else:
                    loc = (0, -1, -1, 0)
                frame = frame[loc[0]:loc[2], loc[3]:loc[1]]
                if self.resize is not None:
                    frame = cv2.resize(frame, self.resize)

                input_frame = self.transform(frame)
                result = self.emotion_recogniser(input_frame.unsqueeze(0).to(self.device))
                logits = result["expression"]
                for ignore_idx in self.ignore_idxs:
                    logits[:, ignore_idx] = float("-1e20")

                val_sequence.append(result["valence"])
                ar_sequence.append(result["arousal"])

                avg_logits += logits
                emotion = self.label_map[np.argmax(logits.detach().cpu().numpy())]

                if emotion not in votes:
                    votes[emotion] = 1
                else:
                    votes[emotion] += 1

                frame_number += 1

        votes_counter = Counter(votes)
        avg_logits /= frame_number
        probabilities = F.softmax(avg_logits, dim=1)

        if self.aggregation == "voting":
            predicted_emotion = votes_counter.most_common(1)[0][0]
        else:
            predicted_emotion = self.label_map[np.argmax(probabilities.detach().cpu().numpy())]

        pred_valence = torch.cat(val_sequence).detach().cpu().numpy()
        pred_arousal = torch.cat(ar_sequence).detach().cpu().numpy()

        if ref_vid is not None and (annotation is None or valence is None or arousal is None):
            ref_video = read_video(ref_vid)
            with torch.no_grad():
                votes = {}
                avg_logits = 0
                frame_number = 0
                val_sequence = []
                ar_sequence = []
                for frame in ref_video:
                    if self.crop:
                        face_loc = face_recognition.face_locations(frame)
                        if face_loc:
                            loc = face_loc[0]
                        else:
                            loc = (0, -1, -1, 0)
                    else:
                        loc = (0, -1, -1, 0)
                    frame = frame[loc[0]:loc[2], loc[3]:loc[1]]
                    if self.resize is not None:
                        frame = cv2.resize(frame, self.resize)

                    input_frame = self.transform(frame)
                    result = self.emotion_recogniser(input_frame.unsqueeze(0).to(self.device))
                    logits = result["expression"]
                    for ignore_idx in self.ignore_idxs:
                        logits[:, ignore_idx] = float("-1e20")

                    val_sequence.append(result["valence"])
                    ar_sequence.append(result["arousal"])
                    avg_logits += logits
                    emotion = self.label_map[np.argmax(logits.detach().cpu().numpy())]

                    if emotion not in votes:
                        votes[emotion] = 1
                    else:
                        votes[emotion] += 1

                    frame_number += 1

            votes_counter = Counter(votes)
            avg_logits /= frame_number
            probabilities = F.softmax(avg_logits, dim=1)

            if annotation is None:
                if self.aggregation == "voting":
                    annotation = votes_counter.most_common(1)[0][0]
                else:
                    annotation = self.label_map[np.argmax(probabilities.detach().cpu().numpy())]

            if valence is None:
                valence = torch.cat(val_sequence).detach().cpu().numpy()

            if arousal is None:
                arousal = torch.cat(ar_sequence).detach().cpu().numpy()

        metrics["accuracy"] = int(annotation == predicted_emotion)
        if valence is not None:
            val_diff = valence - pred_valence
            metrics["valence_difference"] = val_diff.mean()

        if arousal is not None:
            ar_diff = arousal - pred_arousal
            metrics["arousal_difference"] = ar_diff.mean()

        if valence is not None and arousal is not None:
            metrics["emotion_difference"] = np.mean(val_diff ** 2 + ar_diff ** 2)

        return metrics


class MouthEvaluator:
    def __init__(self, lipreader=None, device="cpu", label_map=None):
        alphabet = ['_'] + list(string.ascii_uppercase) + [' ']
        self.ctc_decoder = CtcDecoder(alphabet)
        self.stable_pt_ids = [33, 36, 39, 42, 45]
        self.mouth_pt_ids = range(48, 72)
        self.size = (88, 88)
        self.device = torch.device(device)
        self.mean_face = np.load(os.path.split(__file__)[0] + "/resources/mean_face.npy")

        if lipreader == "lrw":
            model_path = os.path.split(__file__)[0] + "/resources/lrw.pth"
            if not os.path.exists(model_path):
                url = 'https://drive.google.com/uc?id=1oYDAhvYyuFydkECuPJlaDMAl5n_v3jTj'
                gdown.download(url, model_path, quiet=False)

            lipreader = model_path
            label_map = os.path.split(__file__)[0] + "/resources/500WordsSortedList.txt"

        self.label_map = None
        if label_map is not None:
            self.label_map = {}
            with open(label_map) as f:
                annotation_list = f.readlines()
                for l, a in enumerate(annotation_list):
                    self.label_map[l] = a.rstrip()

        if lipreader is None or not os.path.exists(lipreader):
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

    def to(self, device):
        self.device = device
        if self.lipreader is not None:
            self.lipreader.to(device)

    def __call__(self, vid, ref_vid=None, annotation=None, landmarks=None, ref_landmarks=None):
        if (self.lipreader is None or annotation is None) and ref_vid is None:
            warnings.simplefilter("once")
            warnings.warn("You have neither provided a lipreader nor a reference video so there will be no mouth movement evaluation")
            return {}

        if isinstance(landmarks, str):
            landmarks = np.load(landmarks)

        if isinstance(ref_landmarks, str):
            ref_landmarks = np.load(ref_landmarks)

        metrics = {}
        video = read_video(vid)

        if ref_vid is not None:
            ref_video = read_video(ref_vid)
            lmd = 0
            ref_sequence = []

        sequence = []
        no_frames = 0
        for frame in video:
            no_frames += 1
            if landmarks is not None:
                frame_landmarks = landmarks[no_frames - 1, :, :2]
            else:
                try:
                    frame_landmarks = np.array(list(itertools.chain.from_iterable(list(face_recognition.face_landmarks(frame)[0].values()))))
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

            if ref_vid is not None:
                ref_frame = next(ref_video)

                if ref_landmarks is not None:
                    ref_frame_landmarks = ref_landmarks[no_frames - 1, :, :2]
                else:
                    try:
                        ref_frame_landmarks = np.array(list(itertools.chain.from_iterable(list(face_recognition.face_landmarks(ref_frame)[0].values()))))
                        if ref_frame_landmarks is None:
                            return {}
                    except:
                        return {}

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

                # If the frames have different sizes then scale the landmarks accordingly
                scale = ref_frame.shape[1] / frame.shape[1]
                lmd += np.sqrt((ref_frame_landmarks[self.mouth_pt_ids] - scale * frame_landmarks[self.mouth_pt_ids]) ** 2).mean()

        if ref_vid is not None:
            metrics["LMD"] = lmd / no_frames

        if self.lipreader is not None:
            lip_video = torch.cat(sequence)
            with torch.no_grad():
                logits = self.lipreader(lip_video.unsqueeze(0).unsqueeze(0).to(self.device), torch.LongTensor([no_frames]))
                _, predicted = torch.max(F.softmax(logits, dim=1).data, dim=1)
                # With sentence level lip_reading use the following code
                # sentence = self.ctc_decoder.decode_batch(probs.detach().cpu(), torch.LongTensor([probs.size(0)]))[0]
                predicted = predicted.detach().cpu().numpy()[0]
                if self.label_map is not None:
                    predicted = self.label_map[predicted]

            if annotation is None:
                ref_lip_video = torch.cat(ref_sequence)
                warnings.warn("Annotations not provided using reference video prediction as label")
                ref_logits = self.lipreader(ref_lip_video.unsqueeze(0).unsqueeze(0).to(self.device), torch.LongTensor([no_frames]))
                _, gt = torch.max(F.softmax(ref_logits, dim=1).data, dim=1)
                if self.label_map is not None:
                    gt = self.label_map[gt]
            else:
                gt = annotation

            metrics["WER"] = wer(gt.lower(), predicted.lower())

        return metrics
