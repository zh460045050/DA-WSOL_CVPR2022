from torch.autograd import Variable
from sklearn.cluster import k_means
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim
from libs.wsol.losses.mmd import *
from libs.wsol.model.backbone import resnet50, inception_v3, resnet50_adl
from libs.wsol.model.target_assigner import TSA
import math
import cv2
from libs.config import get_configs
from libs.data_loaders import get_data_loader
from libs.inference import CAMComputer
from libs.util import string_contains_any, set_random_seed, color_pro, generate_vis, PerformanceMeter
import libs.wsol
from libs.wsol.model.classifier import CAMClassifier, DomainClassifier, ReverseLayerF, LocalClassifier
from libs.wsol.methods.has import has
from libs.wsol.methods.cutmix import cutmix
import matplotlib.pyplot as plt
from libs.wsol.losses.mmd import cal_mmd

from skimage.segmentation import mark_boundaries
from tensorboardX import SummaryWriter

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224
class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'Top-1 Cls', 'PxAP', 'pIoU', 'GT-Known Loc', 'Top-1 Loc', 'MaxBoxAccV2']
    _BEST_CRITERION_METRIC = 'MaxBoxAccV2'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "OpenImages": 100,
        "ISIC": 3,
        "ILSVRC": 1000,
    }

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                        'Conv2d_3', 'Conv2d_4'],
    }

    def __init__(self):
        self.args = get_configs()
        seed = self.args.seed
        torch.manual_seed(seed)            # 为CPU设置随机种子
        torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        print(self.args.seed)
        
        print(self.args)
        if self.args.architecture == 'inception_v3':
            self.feature_dims = 1024
        else:
            self.feature_dims = 2048

        self.performance_meters = self._set_performance_meters()
        self.reporter = self.args.reporter
        self.tb_writer = SummaryWriter(self.args.log_folder)
        self.extractor, self.classifier = self._set_model()
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.optimizer_extractor, self.optimizer_classifier = self._set_optimizer()

        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class)
        self.lambd = 1
        self.num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]

        self.TSA = TSA(self.num_classes, self.feature_dims, self.args.sample_num_source, self.args.sample_num_target)


        if self.args.uda_method == 'dann':
            #self.mud = MUD(self.num_classes, self.feature_dims, self.args.sample_num_source, self.args.sample_num_target)
            self.domain_classifier = DomainClassifier(num_feature=self.feature_dims)
            self.domain_classifier = self.domain_classifier.cuda()
            self.optimizer_domain_classifier = torch.optim.SGD([
                {'params': self.domain_classifier.parameters(),
                'lr': self.args.lr * self.args.lr_classifier_ratio}],
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                nesterov=True)

    def _set_performance_meters(self):
        self._EVAL_METRICS += ['GT-Known_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]
        self._EVAL_METRICS += ['Top-1_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))

        ###Initialize Extractor
        if self.args.architecture == 'inception_v3':
            extractor = inception_v3(
                        dataset_name=self.args.dataset_name,
                        architecture_type=self.args.architecture_type,
                        pretrained=self.args.pretrained,
                        num_classes=num_classes,
                        large_feature_map=self.args.large_feature_map,
                        pretrained_path=self.args.pretrained_path,
                        adl_drop_rate=self.args.adl_drop_rate,
                        adl_drop_threshold=self.args.adl_threshold,
                        acol_drop_threshold=self.args.acol_threshold,
                        num_head = self.args.num_head)
        elif self.args.architecture == 'resnet50' and self.args.wsol_method == 'adl':
            extractor = resnet50_adl(
                dataset_name=self.args.dataset_name,
                architecture_type=self.args.architecture_type,
                pretrained=self.args.pretrained,
                num_classes=num_classes,
                large_feature_map=self.args.large_feature_map,
                pretrained_path=self.args.pretrained_path,
                adl_drop_rate=self.args.adl_drop_rate,
                adl_drop_threshold=self.args.adl_threshold,
                acol_drop_threshold=self.args.acol_threshold,
                num_head = self.args.num_head)
        else:
            extractor = resnet50(
                dataset_name=self.args.dataset_name,
                architecture_type=self.args.architecture_type,
                pretrained=self.args.pretrained,
                num_classes=num_classes,
                large_feature_map=self.args.large_feature_map,
                pretrained_path=self.args.pretrained_path,
                adl_drop_rate=self.args.adl_drop_rate,
                adl_drop_threshold=self.args.adl_threshold,
                acol_drop_threshold=self.args.acol_threshold,
                num_head = self.args.num_head)
        extractor = extractor.cuda()

        ###Initialize Classifier
        classifier = CAMClassifier(num_classes=num_classes, num_feature=self.feature_dims)
        classifier = classifier.cuda()

        print(extractor, classifier)

        return extractor, classifier

    def _set_optimizer(self):
        param_features = []
        param_classifiers = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        for name, parameter in self.extractor.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.args.architecture)):
                if self.args.architecture in ('inception_v3'):
                    param_features.append(parameter)
                elif 'resnet50' in self.args.architecture:
                    param_classifiers.append(parameter)
            else:
                if self.args.architecture in ('inception_v3'):
                    param_classifiers.append(parameter)
                elif 'resnet50' in self.args.architecture:
                    param_features.append(parameter)

        if self.args.architecture in ('inception_v3'):
            optimizer_extractor = torch.optim.SGD([
                {'params': param_features, 'lr': self.args.lr},
                {'params': param_classifiers,
                'lr': self.args.lr * self.args.lr_classifier_ratio}],
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                nesterov=True)
        elif 'resnet50' in self.args.architecture:
            optimizer_extractor = torch.optim.SGD([
                {'params': param_features, 'lr': self.args.lr},
                {'params': param_classifiers,
                'lr': self.args.lr}],
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                nesterov=True)

        optimizer_classifier = torch.optim.SGD([
            {'params': self.classifier.parameters(),
             'lr': self.args.lr * self.args.lr_classifier_ratio}],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True)
        
        return optimizer_extractor, optimizer_classifier

    def _wsol_training(self, images, target):
        flag = 0

        ###Specific Augmentation for WSOL###
        if (  self.args.wsol_method == 'cutmix' and
                self.args.cutmix_prob > np.random.rand(1) and
                self.args.cutmix_beta > 0):
            #print("!!!!")
            images, target, target_b, phi = cutmix(
                images, target, self.args.cutmix_beta)
            flag = 1
        elif self.args.wsol_method == 'has':
            #print("!!!!")
            images = has(images, self.args.has_grid_size,
                                     self.args.has_drop_rate)

        ###Obtain Source/Target Feature###
        pixel_feature = self.extractor(images)
        image_feature = nn.AdaptiveAvgPool2d(1)(pixel_feature)

        ###Sampling Three subset by TSA###
        if self.args.uda_method != "" and self.epoch > self.args.start_epoch:
            S_T_f, T_t, T_u, _, count, mask = self.TSA.samples_split(image_feature, pixel_feature, target)

        ###Classify the Source Feature###
        logits = self.classifier(image_feature)
        logits = logits.view(images.shape[0], -1)

        ###Classification Loss###
        if self.args.wsol_method == 'cutmix' and flag:
            loss_c = self.cross_entropy_loss(logits, target) * phi  + self.cross_entropy_loss(logits, target_b) * (1-phi)
        else:
            loss_c = self.cross_entropy_loss(logits, target)

        ###Universum Regularization###
        if self.args.uda_method != "" and self.epoch > self.args.start_epoch:
            loss_u = torch.mean( torch.mean(T_u))

        ###Domain Adaption Localization Loss###
        if self.args.uda_method == 'mmd' and self.epoch > self.args.start_epoch:
            
            loss_d = cal_mmd(S_T_f, T_t, count)
            #print(loss_u)
            if self.args.wsol_method == 'cutmix' and flag:
                #print("!!!!")
                S_T_f, T_t, T_u, _, count, mask = self.TSA.samples_split(image_feature, pixel_feature, target_b)
                loss_d = loss_d * phi  + cal_mmd(S_T_f, T_t, count) * (1-phi)
                loss_u = loss_u * phi + (1-phi) * torch.mean( torch.mean(T_u))
            loss = loss_c + self.args.beta * loss_d + self.args.univer * loss_u
        elif self.args.uda_method == 'dann' and self.epoch > self.args.start_epoch:
            
            S_T_f = S_T_f[:count].view(-1, S_T_f.shape[-1], 1, 1)
            S_T_f = ReverseLayerF.apply(S_T_f, self.args.lambd)
            d_logit_source = self.domain_classifier(S_T_f).view(S_T_f.shape[0])

            T_t = T_t[:count].view(-1, T_t.shape[-1], 1, 1)
            T_t = ReverseLayerF.apply(T_t, self.args.lambd)
            d_logit_target = self.domain_classifier(T_t).view(T_t.shape[0])
            loss_d = 0.5 * (nn.BCELoss()(d_logit_source, torch.ones(T_t.shape[0]).cuda()) + nn.BCELoss()(d_logit_target, torch.zeros(T_t.shape[0]).cuda()))
            if self.args.wsol_method == 'cutmix' and flag:
                S_T_f, T_t, T_u, _, count, mask = self.TSA.samples_split(image_feature, pixel_feature, target_b)
                S_T_f = S_T_f[:count].view(-1, S_T_f.shape[-1], 1, 1)
                S_T_f = ReverseLayerF.apply(S_T_f, self.args.lambd)
                d_logit_source = self.domain_classifier(S_T_f).view(S_T_f.shape[0])

                T_t = T_t[:count].view(-1, T_t.shape[-1], 1, 1)
                T_t = ReverseLayerF.apply(T_t, self.args.lambd)
                d_logit_target = self.domain_classifier(T_t).view(T_t.shape[0])
                loss_d = loss_d * phi + (1-phi) * 0.5 * (nn.BCELoss()(d_logit_source, torch.ones(T_t.shape[0]).cuda()) + nn.BCELoss()(d_logit_target, torch.zeros(T_t.shape[0]).cuda()))
                loss_u = loss_u * phi + (1-phi) * torch.mean( torch.mean(T_u))
                
            loss = loss_c + self.args.beta * loss_d + self.args.univer * loss_u
        else:
            loss = loss_c
            loss_d = Variable(torch.zeros(1)).cuda()
            loss_u = Variable(torch.zeros(1)).cuda()
            mask = np.zeros( (pixel_feature.shape[0], pixel_feature.shape[2], pixel_feature.shape[3]) )

        return logits, loss, loss_c, loss_d, loss_u, mask


    def train(self, split):
        self.extractor.train()
        self.classifier.train()
        loader = self.loaders[split]

        total_loss = 0.0
        loss_c_totol = 0.0
        loss_d_totol = 0.0
        loss_u_totol = 0.0
        num_correct = 0
        num_images = 0

        for batch_idx, (images, target, image_ids) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()

            ppp = (self.epoch*len(loader) + batch_idx ) / (self.args.epochs*len(loader))
            self.args.lambd = 2. / (1. + np.exp(-1 * ppp)) - 1

            #print(self.args.lambd)

            if batch_idx % int(len(loader) / 10) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))

            logits, loss, loss_c, loss_d, loss_u, mask = self._wsol_training(images, target)
            #print(loss)
            if self.args.uda_method == 'dann':
                self.optimizer_domain_classifier.zero_grad()

            self.optimizer_extractor.zero_grad()
            self.optimizer_classifier.zero_grad()
            loss.backward()
            self.optimizer_extractor.step()
            self.optimizer_classifier.step()

            if self.args.uda_method == 'dann':
                self.optimizer_domain_classifier.step()
            ####
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)

            loss_c_totol += loss_c.item() * images.size(0)
            loss_d_totol += loss_d.item() * images.size(0)
            loss_u_totol += loss_u.item() * images.size(0)

            num_correct += (pred == target).sum().item()
            num_images += images.size(0)

            ####
            image_size = images.shape[2:]
            image_id = image_ids[0]
            image = images[0, :, :, :]
            cam_path = os.path.join(self.args.log_folder, 'train_mask_vis')
            vis_image = image.cpu().data * np.array(_IMAGENET_STDDEV).reshape([3, 1, 1]) + np.array(_IMAGENET_MEAN).reshape([3, 1, 1])
            vis_image = np.int64(vis_image * 255)
            vis_image[vis_image > 255] = 255
            vis_image[vis_image < 0] = 0
            vis_image = np.uint8(vis_image)
            if not os.path.exists(cam_path):
                os.makedirs(cam_path)
            cam_normalized = cv2.resize(mask[0, :, :], image_size,
                                        interpolation=cv2.INTER_CUBIC)
            plt.imsave(os.path.join(cam_path, str(batch_idx) + ".png"), generate_vis(cam_normalized, vis_image).transpose(1, 2, 0))
            ####

        self.tb_writer.add_scalar('Train Loss/Classification Loss', loss_c_totol / float(num_images), global_step=self.epoch)
        if self.epoch != 0:
            self.tb_writer.add_scalar('Train Loss/Domain Adaption Loss', loss_d_totol / float(num_images), global_step=self.epoch)
            self.tb_writer.add_scalar('Train Loss/Universum Regularization', loss_u_totol / float(num_images), global_step=self.epoch)
        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) * 100

        self.tb_writer.add_scalar('Train Loss/Total Loss', loss_average, global_step=self.epoch)
        self.tb_writer.add_scalar('Train/CLS Acc', classification_acc, global_step=self.epoch)

        self.performance_meters[split]['Top-1 Cls'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def eval_classification(self, split):
        self.extractor.eval()
        self.classifier.eval()
        loader = self.loaders[split]

        total_loss = 0.0
        num_correct = 0
        num_images = 0

        for batch_idx, (images, target, image_ids) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()

            if batch_idx % int(len(loader) / 10) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))
            
            logits, loss, mask = self._wsol_training(images, target)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            num_correct += (pred == target).sum().item()
            num_images += images.size(0)

            for i in range(0, mask.shape[0]):
                ###
                image_id = image_ids[i]
                image = image[i, :, :, :]
                cam_path = ospj(self.log_folder, 'train_mask_vis')
                vis_image = image.cpu().data * np.array(_IMAGENET_STDDEV).reshape([3, 1, 1]) + np.array(_IMAGENET_MEAN).reshape([3, 1, 1])
                vis_image = np.int64(vis_image * 255)
                vis_image[vis_image > 255] = 255
                vis_image[vis_image < 0] = 0
                vis_image = np.uint8(vis_image)
                if not os.path.exists(cam_path):
                    os.makedirs(cam_path)
                plt.imsave(os.path.join(cam_path, image_id + ".png"), generate_vis(cam_normalized, vis_image).transpose(1, 2, 0))
                ###

            #self.optimizer.zero_grad()
            #loss.backward()
            #self.optimizer.step()

        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) * 100

        self.performance_meters[split]['Top-1 Cls'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def print_performances(self):
        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    print("Split {}, metric {}, current value: {}".format(
                        split, metric, current_performance))
                    if split != 'test':
                        print("Split {}, metric {}, best value: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_value))
                        print("Split {}, metric {}, best epoch: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_epoch))

    def save_performances(self):
        log_path = os.path.join(self.args.log_folder, 'performance_log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.performance_meters, f)

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0

        for i, (images, targets, image_ids) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda()
            pixel_feature = self.extractor(images)
            image_feature = nn.AdaptiveAvgPool2d(1)(pixel_feature)
            logits = self.classifier(image_feature)
            logits = logits.view(images.shape[0], -1)
            pred = logits.argmax(dim=1)
            
            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100
        return classification_acc

    def evaluate(self, epoch, split):
        print("Evaluate epoch {}, split {}".format(epoch, split))
        self.extractor.eval()
        self.classifier.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['Top-1 Cls'].update(accuracy)

        cam_computer = CAMComputer(
            extractor=self.extractor,
            classifier=self.classifier,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset_name,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            log_folder=self.args.log_folder,
            wsol_method = self.args.wsol_method,
        )
        cam_performance = cam_computer.compute_and_evaluate_cams()
        #_EVAL_METRICS = ['loss', 'Top-1 Cls', 'PxAP', 'pIoU', 'GT-Known Loc', 'Top-1 Loc', 'MaxBoxAccV2']

        if self.args.dataset_name == 'OpenImages' or self.args.dataset_name == 'ISIC':
            pxap = np.average(cam_performance['pxap'])
            piou = np.max(cam_performance['iou'])
            self.performance_meters[split]['PxAP'].update(pxap)
            self.performance_meters[split]['pIoU'].update(piou)
            localization = pxap

        elif self.args.dataset_name == 'CUB':
            if split == "test":
                pxap = np.average(cam_performance['pxap'])
                piou = np.max(cam_performance['iou'])
                self.performance_meters[split]['PxAP'].update(pxap)
                self.performance_meters[split]['pIoU'].update(piou)
            gt_known = cam_performance['gt_known'][self.args.iou_threshold_list.index(50)]
            top_1 = cam_performance['top_1'][self.args.iou_threshold_list.index(50)]
            maxboxacc = np.average(cam_performance['gt_known'])
            self.performance_meters[split]['Top-1 Loc'].update(top_1)
            self.performance_meters[split]['GT-Known Loc'].update(gt_known)
            self.performance_meters[split]['MaxBoxAccV2'].update(maxboxacc)
            localization = maxboxacc
            
        else:
            gt_known = cam_performance['gt_known'][self.args.iou_threshold_list.index(50)]
            top_1 = cam_performance['top_1'][self.args.iou_threshold_list.index(50)]
            maxboxacc = np.average(cam_performance['gt_known'])
            self.performance_meters[split]['Top-1 Loc'].update(top_1)
            self.performance_meters[split]['GT-Known Loc'].update(gt_known)
            self.performance_meters[split]['MaxBoxAccV2'].update(maxboxacc)
            localization = maxboxacc
    
        if split == 'val':
            self.tb_writer.add_scalar('Val/CLS Acc', accuracy, global_step=self.epoch)
            self.tb_writer.add_scalar('Val/MaxBoxAccV2', localization, global_step=self.epoch)

        #if self.args.dataset_name in ('CUB', 'ILSVRC'):
        #    for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
        #        self.performance_meters[split][
        #            'GT-Konwn_IOU_{}'.format(IOU_THRESHOLD)].update(
        #            cam_performance['gt_known'][idx])
        #        self.performance_meters[split][
        #            'Top-1_IOU_{}'.format(IOU_THRESHOLD)].update(
        #            cam_performance['top_1'][idx])

    def _torch_save_model(self, filename, epoch):
        if self.args.wsol_method == 'dann':
            torch.save({'architecture': self.args.architecture,
                        'epoch': epoch,
                        'state_dict_extractor': self.extractor.state_dict(),
                        'optimizer_extractor': self.optimizer_extractor.state_dict(),
                        'state_dict_classifier': self.classifier.state_dict(),
                        'optimizer_classifier': self.optimizer_classifier.state_dict(),
                        'state_dict_domain_classifier': self.domain_classifier.state_dict(), 
                        'optimizer_domain_classifier': self.optimizer_domain_classifier.state_dict()},
                        os.path.join(self.args.log_folder, filename))
        else:
            torch.save({'architecture': self.args.architecture,
            'epoch': epoch,
            'state_dict_extractor': self.extractor.state_dict(),
            'optimizer_extractor': self.optimizer_extractor.state_dict(),
            'state_dict_classifier': self.classifier.state_dict(),
            'optimizer_classifier': self.optimizer_classifier.state_dict()},
            os.path.join(self.args.log_folder, filename))

    def save_checkpoint(self, epoch, split):
        self._torch_save_model(
            self._CHECKPOINT_NAME_TEMPLATE.format('%d'%(epoch)), epoch)
        print("saving checkpoint:", self._CHECKPOINT_NAME_TEMPLATE.format('%d'%(epoch)))
        if self.args.epochs == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('last'), epoch)
            print("saving checkpoint:", self._CHECKPOINT_NAME_TEMPLATE.format('last'))

    def report_train(self, train_performance, epoch, split='train'):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        reporter_instance.add(
            key='{split}/Top-1 Cls'.format(split=split),
            val=self.performance_meters['train']['Top-1 Cls'].current_value)
        reporter_instance.add(
            key='{split}/loss'.format(split=split),
            val=self.performance_meters['train']['loss'].current_value)
        reporter_instance.write()

    def report(self, epoch, split):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        for metric in self._EVAL_METRICS:
            reporter_instance.add(
                key='{split}/{metric}'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].current_value)
            reporter_instance.add(
                key='{split}/{metric}_best'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].best_value)
        reporter_instance.write()

    def adjust_learning_rate(self, epoch):
        if epoch != 0 and epoch % self.args.lr_decay_frequency == 0:
            for param_group in self.optimizer_extractor.param_groups:
                param_group['lr'] *= 0.1
            for param_group in self.optimizer_classifier.param_groups:
                param_group['lr'] *= 0.1

    def load_checkpoint(self, checkpoint_type):
        if checkpoint_type not in ('best', 'last'):
            raise ValueError("checkpoint_type must be either best or last.")
        checkpoint_path = os.path.join(
            self.args.log_folder,
            self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.extractor.load_state_dict(checkpoint['state_dict_extractor'], strict=True)
            self.classifier.load_state_dict(checkpoint['state_dict_classifier'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))
