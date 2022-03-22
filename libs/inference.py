"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cv2
import numpy as np
import os
from os.path import join as ospj
from os.path import dirname as ospd
import torch.nn as nn
from libs.evaluation import BoxEvaluator
from libs.evaluation import MaskEvaluator
from libs.evaluation import configure_metadata
from libs.evaluation import MultiClassEvaluator
from libs.util import t2n
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries

##
from libs.bps.bp import BackPropagation
from libs.bps.gbp import GuidedBackPropagation
from libs.bps.grad_cam import GradCAM
from libs.bps.grad_cam_pp import GradCAMpp
from libs.bps.dynamic_bp import DynamicBP 
from libs.bps.pcs import PCS
from libs.bps.bagcams import BagCAMs
from libs.bps.cbm_ablation import CBM_Ablation
from libs.bps.cbm_pp import CBM_pp
from libs.bps.dynamic_cbm import Dynamic_CBM
from libs.bps.cic import CIC

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def generate_vis(p, img):
    # All the input should be numpy.array 
    # img should be 0-255 uint8

    C = 1
    H, W = p.shape

    prob = p

    prob[prob<=0] = 1e-7

    def ColorCAM(prob, img):
        C = 1
        H, W = prob.shape
        colorlist = []
        colorlist.append(color_pro(prob,img=img,mode='chw'))
        CAM = np.array(colorlist)/255.0
        return CAM

    #print(prob.shape, img.shape)
    CAM = ColorCAM(prob, img)
    #print(CAM.shape)
    return CAM[0, :, :, :]

def color_pro(pro, img=None, mode='hwc'):
	H, W = pro.shape
	pro_255 = (pro*255).astype(np.uint8)
	pro_255 = np.expand_dims(pro_255,axis=2)
	color = cv2.applyColorMap(pro_255,cv2.COLORMAP_JET)
	color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
	if img is not None:
		rate = 0.5
		if mode == 'hwc':
			assert img.shape[0] == H and img.shape[1] == W
			color = cv2.addWeighted(img,rate,color,1-rate,0)
		elif mode == 'chw':
			assert img.shape[1] == H and img.shape[2] == W
			img = np.transpose(img,(1,2,0))
			color = cv2.addWeighted(img,rate,color,1-rate,0)
			color = np.transpose(color,(2,0,1))
	else:
		if mode == 'chw':
			color = np.transpose(color,(2,0,1))	
	return color

def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


class CAMComputer(object):
    def __init__(self, extractor, classifier, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, log_folder=None, wsol_method='cam', post_method='cam', target_layer='layer3', is_vis=False):
        self.extractor = extractor
        self.classifier = classifier
        self.extractor.eval()
        self.classifier.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder
        self.vis = is_vis
        #
        self.wsol_method = wsol_method
        self.post_method = post_method
        self.target_layer = target_layer
        #
        if self.post_method == "dbp":
            self.gcam = DynamicBP(extractor=self.extractor, classifier=self.classifier)
        elif self.post_method == 'bp':
            self.gcam = BackPropagation(extractor=self.extractor, classifier=self.classifier)
        elif self.post_method == 'gbp':
            self.gcam = GuidedBackPropagation(extractor=self.extractor, classifier=self.classifier)
        elif self.post_method == 'GradCAM':
            self.gcam = GradCAM(extractor=self.extractor, classifier=self.classifier)
        elif self.post_method == 'GradCAM++':
            self.gcam = GradCAMpp(extractor=self.extractor, classifier=self.classifier)
        elif self.post_method == 'PCS':
            self.gcam = PCS(extractor=self.extractor, classifier=self.classifier)
        elif self.post_method == 'BagCAMs':
            self.gcam = BagCAMs(extractor=self.extractor, classifier=self.classifier)
        elif self.post_method == 'cbm_ablation':
            self.gcam = CBM_Ablation(extractor=self.extractor, classifier=self.classifier)
        elif self.post_method == 'cbm++':
            self.gcam = CBM_pp(extractor=self.extractor, classifier=self.classifier)
        elif self.post_method == 'dcbm':
            self.gcam = Dynamic_CBM(extractor=self.extractor, classifier=self.classifier)
        elif self.post_method == 'cic':
            self.gcam = CIC(extractor=self.extractor, classifier=self.classifier)

        if dataset_name == "CUB":
            metadata_mask = configure_metadata('metadata/CUBMask/' + split)
        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))
        self.dataset_name = dataset_name

        if dataset_name == "ILSVRC":
            self.evaluator_boxes = {
                            "ILSVRC": BoxEvaluator,
                            }[dataset_name](metadata=metadata,
                                            dataset_name=dataset_name,
                                            split=split,
                                            cam_threshold_list=cam_threshold_list,
                                            iou_threshold_list=iou_threshold_list,
                                            mask_root=mask_root,
                                            multi_contour_eval=multi_contour_eval,
                                            scoremap_root=ospj(self.log_folder, 'scoremaps'))
        elif dataset_name == "CUB":
            self.evaluator_boxes = BoxEvaluator(metadata=metadata,
                            dataset_name="CUB",
                            split=split,
                            cam_threshold_list=cam_threshold_list,
                            iou_threshold_list=iou_threshold_list,
                            mask_root=mask_root,
                            multi_contour_eval=multi_contour_eval,
                            scoremap_root=ospj(self.log_folder, 'scoremaps'))
            if self.split == "test":
                self.evaluator_mask = MaskEvaluator(metadata=metadata_mask,
                                    dataset_name="CUBMask",
                                    split=split,
                                    cam_threshold_list=cam_threshold_list,
                                    iou_threshold_list=iou_threshold_list,
                                    mask_root=mask_root,
                                    multi_contour_eval=multi_contour_eval,
                                    scoremap_root=ospj(self.log_folder, 'scoremaps'))
        elif self.dataset_name == "VOC":
            self.evaluator_mask = MultiClassEvaluator(metadata=metadata,
                                            dataset_name=dataset_name,
                                            split=split,
                                            cam_threshold_list=cam_threshold_list,
                                            iou_threshold_list=iou_threshold_list,
                                            mask_root=mask_root,
                                            multi_contour_eval=multi_contour_eval,
                                            scoremap_root=ospj(self.log_folder, 'scoremaps'))
        else:
            self.evaluator_mask = MaskEvaluator(metadata=metadata,
                                            dataset_name=dataset_name,
                                            split=split,
                                            cam_threshold_list=cam_threshold_list,
                                            iou_threshold_list=iou_threshold_list,
                                            mask_root=mask_root,
                                            multi_contour_eval=multi_contour_eval,
                                            scoremap_root=ospj(self.log_folder, 'scoremaps'))


    def compute_and_evaluate_cams(self):

        print("Computing and evaluating %s ."%(self.post_method))
        for images, targets, image_ids in self.loader:
            image_size = images.shape[2:]
            images = images.cuda()
            targets = targets.cuda()
        
            if self.post_method == 'cam':
                pixel_features = self.extractor(images)
                cams = self.classifier(pixel_features, targets, return_cam=True)
                if len(targets.shape) == 2:
                    cams = F.interpolate(
                        cams, image_size, mode="bilinear", align_corners=False
                    )
                    from libs.wsol.util import normalize_tensor
                    cams = normalize_tensor(cams.detach().clone())
                cams = t2n(cams)

                image_features = nn.AdaptiveAvgPool2d(1)(pixel_features)
                logits = self.classifier(image_features)
            else:
                logits, probs = self.gcam.forward(images)
                #predicts = torch.argmax(logits, dim=1)
                _, ids = probs.sort(dim=1, descending=True)
                #print(ids.shape, targets.shape)
                self.gcam.backward(ids=targets)
                if self.target_layer == "":
                    #cams = self.gcam.generate_all_layer().squeeze(1)
                    cams = t2n(self.gcam.generate_all_layer().squeeze(1))
                else:
                    cams = t2n(self.gcam.generate(target_layer=self.target_layer).squeeze(1))
                    #cams = self.gcam.generate(target_layer=self.target_layer).squeeze(1)
            predicts = torch.argmax(logits, dim=1)
            
            predicts = t2n(predicts.view(predicts.shape[0], 1))
            targets = t2n(targets)
            
            #print(targets)

            for cam, target, predict, image, image_id in zip(cams, targets, predicts, images, image_ids):
                #print(aaa)
                #cam = t2n(cam)
                #predict = t2n(predict)
                #target = t2n(target)
                if len(targets.shape) != 2:
                    #print("!!!")
                    cam_resized = cv2.resize(cam, image_size,
                                            interpolation=cv2.INTER_CUBIC)
                    #print(cam_resized.shape)
                    cam_normalized = normalize_scoremap(cam_resized)
                else:
                    cam_normalized = cam
                if self.split in ('val', 'test'):
                    cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                    pred_path = ospj(self.log_folder, 'predicts', image_id)
                    gt_path = ospj(self.log_folder, 'image_gts', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    if not os.path.exists(ospd(pred_path)):
                        os.makedirs(ospd(pred_path))
                    if not os.path.exists(ospd(gt_path)):
                        os.makedirs(ospd(gt_path))
                    np.save(ospj(cam_path), cam_normalized)
                    np.save(ospj(pred_path), predict)
                    np.save(ospj(gt_path), target)
                    
                    ###
                    if not (self.dataset_name == "ILSVRC" and self.split == "val"): 
                        if self.vis:
                            vis_image = image.cpu().data * np.array(_IMAGENET_STDDEV).reshape([3, 1, 1]) + np.array(_IMAGENET_MEAN).reshape([3, 1, 1])
                            vis_image = np.int64(vis_image * 255)
                            vis_image[vis_image > 255] = 255
                            vis_image[vis_image < 0] = 0
                            vis_image = np.uint8(vis_image)
                            vis_path = ospj(self.log_folder, 'vis', image_id)
                            #print(target.shape, cam_normalized.shape)
                            if not os.path.exists(ospd(vis_path)):
                                os.makedirs(ospd(vis_path))
                            if len(targets.shape) == 2:
                                for i, flag in enumerate(target):
                                    if flag:
                                        plt.imsave(ospj(vis_path[:-4])+ "_" + str(i) + ".png", generate_vis(cam_normalized[i], vis_image).transpose(1, 2, 0))
                            else:
                                plt.imsave(ospj(vis_path)+".png", generate_vis(cam_normalized, vis_image).transpose(1, 2, 0))
                    ###
                    
            #np.save(ospj(cam_path), cam_normalized)
            performance={}
            if self.dataset_name == "ILSVRC":
                self.evaluator_boxes.accumulate(cam_normalized, image_id)
                gt_known = self.evaluator_boxes.compute()
                top_1 = self.evaluator_boxes.compute_top1()
                performance['gt_known'] = gt_known
                performance['top_1'] = top_1

            elif self.dataset_name == "CUB":
                if self.split == "test":
                    self.evaluator_mask.accumulate(cam_normalized, image_id)
                    pxap, iou = self.evaluator_mask.compute()
                    performance['pxap'] = pxap
                    performance['iou'] = iou

                self.evaluator_boxes.accumulate(cam_normalized, image_id)
                gt_known = self.evaluator_boxes.compute()
                top_1 = self.evaluator_boxes.compute_top1()
                performance['gt_known'] = gt_known
                performance['top_1'] = top_1
                
            else:
                self.evaluator_mask.accumulate(cam_normalized, image_id)
                pxap, iou = self.evaluator_mask.compute()
                performance['pxap'] = pxap
                performance['iou'] = iou

        return performance
