from torch.autograd import Variable
from sklearn.cluster import k_means
import numpy as np
import torch
import random

class TSA(object):

    def __init__(self, num_classes, feature_dims, sample_num_source=32, sample_num_target=32):
        self.num_classes = num_classes 
        self.feature_dims = feature_dims
        self.cluster_centers = np.zeros( (self.num_classes, self.feature_dims) ) #M_{:,1:}
        self.cluster_counts = np.zeros( (self.num_classes, 1) ) #r_{1:}
        self.universe_centers = np.zeros( (1, self.feature_dims) ) #M_{:,0}
        self.universe_count = 0 #r_{0}
        self.sample_num_source = sample_num_source
        self.sample_num_target = sample_num_target

    def samples_split(self, image_feature, pixel_feature, target):

        source = image_feature.view(-1, image_feature.shape[1])
        mask = np.zeros( (source.shape[0], pixel_feature.shape[2], pixel_feature.shape[3]) )
        batch_source = torch.zeros(source.shape[0], self.sample_num_source, image_feature.shape[1]).cuda()
        batch_target = torch.zeros(source.shape[0], self.sample_num_target, image_feature.shape[1]).cuda()
        batch_label = torch.zeros(source.shape[0]).cuda()
        count = 0
        count_u = 0
        for i in range(0, source.shape[0]):
            batch_source[count, 0, :] = source[i, :]
            lab = target[i]
            all_samples = pixel_feature.permute(0, 2, 3, 1).contiguous()[i, :, :, :].view(-1, source.shape[1])
            samples = all_samples.detach().clone().cpu().numpy()

            if self.cluster_counts[lab] == 0:
                self.cluster_counts[lab] = self.cluster_counts[lab] + 1
                self.cluster_centers[lab, :] = source[i, :].unsqueeze(0).detach().clone().cpu().numpy() / self.cluster_counts[lab] + self.cluster_centers[lab, :] * ((self.cluster_counts[lab]-1) / self.cluster_counts[lab] )

            #### Kmeans Cluster
            center_inits = torch.cat( [torch.from_numpy(self.universe_centers).cuda(), source[i, :].unsqueeze(0), torch.from_numpy(self.cluster_centers)[lab, :].unsqueeze(0).cuda()], dim=0).detach().clone().cpu().numpy()
            center, label, pb = k_means(samples, n_clusters=3, init=center_inits, n_init=1, random_state=0)

            #### Update Cache Matrix
            self.cluster_counts[lab] = self.cluster_counts[lab] + 1
            self.cluster_centers[lab, :] = np.expand_dims(center[1, :], axis=0) / self.cluster_counts[lab] + self.cluster_centers[lab, :] * ((self.cluster_counts[lab] - 1) / self.cluster_counts[lab] )

            self.universe_count = self.universe_count + 1
            self.universe_centers[0, :] = np.expand_dims(center[0, :], axis=0) / self.universe_count + self.universe_centers[0, :] * ((self.universe_count - 1) / self.universe_count )

            #### Sample Source/Target/Univers Items
            cur_univer = all_samples[label == 0, :]
            cur_source = all_samples[label == 1, :]
            cur_target = all_samples[label == 2, :]

            #### Random Sampling
            rand_index_target = np.arange(0, cur_target.shape[0])
            random.shuffle(rand_index_target)
            rand_index_source = np.arange(0, cur_source.shape[0])
            random.shuffle(rand_index_source)

            if len(rand_index_target) >= self.sample_num_target and len(rand_index_source) >= (self.sample_num_source - 1):
                batch_source[count, 1:, :] = torch.index_select(cur_source, 0, torch.from_numpy(rand_index_source)[:min(self.sample_num_source-1, len(rand_index_source))].cuda())
                batch_target[count, :, :] = torch.index_select(cur_target, 0, torch.from_numpy(rand_index_target)[:min(self.sample_num_target, len(rand_index_target))].cuda())
                batch_label[count] = target[i]
                count = count + 1
            if count_u == 0:
                batch_universum = cur_univer.view(-1, image_feature.shape[1])
                count_u = batch_universum.shape[1]
            else:
                batch_universum = torch.cat( [batch_universum, cur_univer.view(-1, image_feature.shape[1])], dim=0)

            label = np.float32(label)
            label[label == 2] = 0.5
            mask[i, :, :] = label.reshape( (pixel_feature.shape[2], pixel_feature.shape[3]) )

        #print(count)
        return batch_source, batch_target, batch_universum, batch_label, count, mask
