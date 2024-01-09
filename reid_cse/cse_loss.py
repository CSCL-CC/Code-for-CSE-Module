import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
import numpy as np
import pickle
import cv2
import os.path as osp


class ContinuousSurfaceEmbeddingLoss(nn.Module):
    def __init__(self, embedding_dim,
                 embdist_gauss_sigma=1,
                 heatmap_gauss_sigma=128,
                 weight_dp_masks=1.0,
                 weight_cse=1.0,
                 n_vertex=27554,
                 feature_dim=256):
        super(ContinuousSurfaceEmbeddingLoss, self).__init__()
        self.weight_dp_masks = weight_dp_masks
        self.weight_cse = weight_cse
        self.features = nn.Parameter(torch.rand(n_vertex, feature_dim) * 0.02)
        self.proj = nn.Parameter(torch.rand(feature_dim, embedding_dim) * 0.02)
        self.embedding_dim = embedding_dim
        self.embdist_gauss_sigma = nn.Parameter(torch.ones([1] * embdist_gauss_sigma))
        self.heatmap_gauss_sigma = heatmap_gauss_sigma
        self._loadEval()
        self.vertex = set()
        self.i = 0

    def _loadEval(self):
        base_path = './data/DP3D/'
        smpl_subdiv_fpath = osp.join(base_path, 'SMPL_subdiv.mat')
        SMPL_subdiv = loadmat(smpl_subdiv_fpath)
        UV = np.array([SMPL_subdiv["U_subdiv"], SMPL_subdiv["V_subdiv"]]).squeeze()
        ClosestVertsInds = np.arange(UV.shape[1]) + 1
        self.Part_UVs = []
        self.Part_ClosestVertInds = []
        for i in np.arange(24):
            self.Part_UVs.append(
                torch.from_numpy(UV[:, SMPL_subdiv['Part_ID_subdiv'].squeeze() == (i + 1)]).cuda()
            )
            self.Part_ClosestVertInds.append(
                torch.from_numpy(ClosestVertsInds[SMPL_subdiv["Part_ID_subdiv"].squeeze() == (i + 1)]).cuda()
            )
        pdist_transform_fpath = osp.join(base_path, 'SMPL_SUBDIV_TRANSFORM.mat')
        self.PDIST_transform = loadmat(pdist_transform_fpath)
        self.PDIST_transform = torch.from_numpy(self.PDIST_transform["index"].squeeze().astype(np.int32)).cuda().long()
        geodists = osp.join(base_path, 'geodists_smpl_27554.pkl')
        with open(geodists, 'rb') as hFile:
            self.geodists = torch.from_numpy(pickle.load(hFile)).float()
        self.Mean_Distances = torch.Tensor([0, 0.351, 0.107, 0.126, 0.237, 0.173, 0.142, 0.128, 0.150]).cuda()
        self.CoarseParts = torch.Tensor(
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8]).cuda().long()
        self.Part_ids = torch.Tensor(SMPL_subdiv['Part_ID_subdiv']).squeeze().cuda().long()
        self.gpsm = []

    def accumulate(self):
        min_threshold = 0.5
        iouThrs = np.linspace(min_threshold, 0.95, int(np.round((0.95 - min_threshold) / 0.05)) + 1, endpoint=True)
        gpsm = torch.cat(self.gpsm, 0)
        instance_num = torch.Tensor([gpsm.size(0)]).cuda()
        AP = torch.Tensor([(gpsm > t).float().sum() for t in iouThrs]).cuda()
        # torch.distributed.all_reduce(instance_num)
        # torch.distributed.all_reduce(AP)
        mAP = (AP / instance_num).mean()
        self.gpsm = []
        return mAP

    def squared_euclidean_distance_matrix(self, pts1, pts2):
        edm = torch.mm(-2 * pts1, pts2.t())
        edm += torch.sum(pts1 * pts1, dim=1, keepdim=True) + torch.sum(pts2 * pts2, dim=1, keepdim=True).t()
        return edm.contiguous()

    def findAllClosestVerts(self, dp_I, dp_U, dp_V):
        ClosestVertsGT = torch.ones_like(dp_I).long() * -1
        for i in range(24):
            index = dp_I == (i + 1)
            if index.sum() == 0:
                continue
            UVs = torch.stack([dp_U[index], dp_V[index]], 1)
            Current_Part_UVs = self.Part_UVs[i].permute(1, 0)
            Current_Part_ClosestVertInds = self.Part_ClosestVertInds[i]
            D = self.squared_euclidean_distance_matrix(UVs, Current_Part_UVs.float())
            ClosestVertsGT[dp_I == (i + 1)] = Current_Part_ClosestVertInds[D.argmin(dim=1)] - 1
        ClosestVertsGT = self.PDIST_transform[ClosestVertsGT] - 1
        return ClosestVertsGT

    def interpolate_vertex_embedding(self, embeddings, dp_index, dp_x, dp_y):
        floor_x, ceil_x = torch.floor(dp_x).long(), torch.ceil(dp_x).long()
        floor_y, ceil_y = torch.floor(dp_y).long(), torch.ceil(dp_y).long()
        # along x
        embed_y1 = torch.where((ceil_x == floor_x)[:, None].expand(-1, self.embedding_dim),
                               embeddings[dp_index, :, floor_y, ceil_x],
                               embeddings[dp_index, :, floor_y, ceil_x] * (ceil_x - dp_x)[:, None] +
                               embeddings[dp_index, :, floor_y, floor_x] * (dp_x - floor_x)[:, None]
                               )
        embed_y2 = torch.where((ceil_x == floor_x)[:, None].expand(-1, self.embedding_dim),
                               embeddings[dp_index, :, ceil_y, ceil_x],
                               embeddings[dp_index, :, ceil_y, ceil_x] * (ceil_x - dp_x)[:, None] +
                               embeddings[dp_index, :, ceil_y, floor_x] * (dp_x - floor_x)[:, None]
                               )
        # along y
        vertex_embeddings = torch.where(
            (ceil_y == floor_y)[:, None].expand(-1, self.embedding_dim),
            embed_y1,
            embed_y2 * (ceil_y - dp_y)[:, None] + embed_y1 * (dp_y - floor_y)[:, None]
        )
        return vertex_embeddings

    def forward(self, dp_masks_pred, dp_masks_gt, cse_pred, dp_x, dp_y, dp_I, dp_U,
                dp_V, rgb_img, evaluate=False):
        self.embdist_gauss_sigma.data = self.embdist_gauss_sigma.data.clamp_(0, 4.85)
        losses = dict()
        if dp_masks_pred.size(1) == 1:
            dp_masks_gt = (dp_masks_gt != 0).unsqueeze(1)
            losses['loss_dp_masks'] = torch.nn.BCEWithLogitsLoss()(dp_masks_pred,
                                                                   dp_masks_gt.float()) * self.weight_dp_masks
            dp_masks_pred, dp_masks_gt.float() * self.weight_dp_masks
        else:
            losses['loss_dp_masks'] = F.cross_entropy(dp_masks_pred, dp_masks_gt.long()) * self.weight_dp_masks
        n, _, h, w = cse_pred.size()
        if self.i % 20 == 0:
            with torch.no_grad():
                img = cse_pred[0, 0:4].float()
                img = ((img + 1) / 2.0 * 255).round()
                mask = (dp_masks_pred[0].sigmoid() > 0.5).float().expand(img.size(0), -1, -1)
                gt_mask = dp_masks_gt[0].expand(img.size(0), -1, -1)
                gt_img = img.clone()
                img[mask == 0] = 0
                gt_img[gt_mask == 0] = 0
                mean = torch.Tensor([102.9801, 115.9465, 122.7717])[:, None, None].cuda()
                std = torch.Tensor([58.8235294117647, 58.8235294117647, 58.8235294117647])[:, None, None].cuda()
                rgb_img = (rgb_img[0] * std) + mean
                rgb_img[:, dp_y[0].round().long(), dp_x[0].round().long()] = torch.Tensor([0, 0, 255]).cuda().unsqueeze(
                    1)
                rgb_img = torch.cat([rgb_img, 255 * torch.ones([1, h, w]).cuda()], 0)
                img = torch.cat([rgb_img, gt_img, img], 2)
                img = img.permute(1, 2, 0).cpu().numpy()
        self.i += 1
        split_index = [x.size(0) for x in dp_x]
        dp_index = [torch.ones(split_index[i]).cuda() * i for i in range(len(split_index))]
        dp_index = torch.cat(dp_index, 0).long()
        dp_x = torch.cat(dp_x, 0).cuda()
        dp_y = torch.cat(dp_y, 0).cuda()
        dp_I = torch.cat(dp_I, 0).cuda()
        dp_U = torch.cat(dp_U, 0).cuda()
        dp_V = torch.cat(dp_V, 0).cuda()
        vertex_embeddings = self.interpolate_vertex_embedding(cse_pred, dp_index, dp_x, dp_y)
        vertex_indices = self.findAllClosestVerts(dp_I, dp_U, dp_V)
        cosine = torch.matmul(
            F.normalize(vertex_embeddings, dim=-1),
            F.normalize(torch.matmul(self.features, self.proj), dim=-1).permute(1, 0))
        if evaluate:
            # compute GPS
            vertex_indices_gt = vertex_indices
            vertex_indices_pred = cosine.argmax(1)
            dist = self.geodists[vertex_indices_gt.cpu(), vertex_indices_pred.cpu()]
            current_mean_distances = self.Mean_Distances[self.CoarseParts[self.Part_ids[vertex_indices_gt]]].cpu()
            ogps_values = torch.exp(-(dist ** 2) / (2 * (current_mean_distances ** 2)))
            ogps_values = torch.split(ogps_values, split_index, dim=0)
            ogps_values = [(x.sum() / x.size(0)).view(1) if x.size(0) != 0 else torch.Tensor([1]) for x in
                           ogps_values]
            ogps = torch.cat(ogps_values).cuda()
            ious = ((dp_masks_pred.sigmoid() > 0.5) & (dp_masks_gt != 0)).float().sum([1, 2, 3]) / (
                    (dp_masks_pred.sigmoid() > 0.5) | (dp_masks_gt != 0)).float().sum([1, 2, 3])
            self.gpsm.append((ious * ogps).sqrt())
            return

        logit = cosine * torch.exp(self.embdist_gauss_sigma)
        embdist_logsoftmax_values = F.log_softmax(logit, dim=1)
        with torch.no_grad():
            geodist_softmax_values = F.softmax(-self.geodists[vertex_indices.cpu()].cuda() * self.heatmap_gauss_sigma,
                                               dim=1)
        losses['loss_cse'] = (-geodist_softmax_values * embdist_logsoftmax_values).sum(1).mean() * self.weight_cse
        acc = (cosine.topk(k=10, dim=1)[1] == vertex_indices[:, None]).float().sum(1).mean()
        return losses, acc

if __name__ == '__main__':
    loss = ContinuousSurfaceEmbeddingLoss(64)
    data = torch.rand(())
