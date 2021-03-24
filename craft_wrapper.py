from CRAFT_pytorch.craft import CRAFT
from CRAFT_pytorch import imgproc, craft_utils
from collections import OrderedDict
from torch.autograd import Variable
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

class CRAFT_pytorch:
    def __init__(self, trained_model='craft_mlt_25k.pth',
             cuda=torch.cuda.is_available(),
             refine=False,
             refiner_model='craft_refiner_CTW1500.pth'):
        # Load net
        self.net = CRAFT()
        self.cuda = False
        print('Loading weights from checkpoint (' + trained_model + ')')
        if cuda:
            self.net.load_state_dict(copyStateDict(torch.load(trained_model)))
            self.cuda = True
        else:
            self.net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

        if cuda:
            self.net = self.net.cuda()
            cudnn.benchmark = False

        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if refine:
            from CRAFT_pytorch.refinenet import RefineNet
            self.refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
            if cuda:
                self.refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
                self.refine_net = self.refine_net.cuda()
            else:
                self.refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))

            self.refine_net.eval()
            self.poly = True

    def detect_text(self,
                    image,
                    text_threshold=0.7,
                    low_text=0.4,
                    link_threshold=0.4,
                    canvas_size=1280,
                    mag_ratio=1.5,
                    poly=False):
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        return boxes, polys, ret_score_text