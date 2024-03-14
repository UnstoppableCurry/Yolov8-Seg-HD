# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist
from concurrent.futures import ThreadPoolExecutor


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # masks loss
            # masks = batch['masks'].to(self.device).float()
            mask = batch['mask'].to(self.device).float()

            # if isinstance(batch['mask'],list):
            #     mask = batch['mask'].to(self.device).float()
            # else:
            #     mask = batch['masks'].to(self.device).float()
            if tuple(mask.shape[-2:]) != (mask_h, mask_w):  # downsample
                mask = F.interpolate(mask[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(mask[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = mask[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss_GPU(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy,
                                                         marea, mask[i], batch['img'][i])  # seg

                # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
                else:
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box / batch_size  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def findContours(self, mask):
        # 将掩码转换为二值图像
        binary = (mask > 0).float()

        # 提取图像边缘
        edges = (torch.nn.functional.conv2d(binary.unsqueeze(0), torch.ones(1, 1, 3, 3),
                                            padding=1) < 9).float()

        # 找到边缘像素的坐标
        contours = torch.nonzero(edges.squeeze(), as_tuple=False)

        return contours

    def torch2D_Hausdorff_distance(self, x, y):  # Input be like (Batch,width,height)
        x = x.float()
        y = y.float()
        distance_matrix = torch.cdist(x, y, p=2)  # p=2 means Euclidean Distance

        value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
        value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

        value = torch.cat((value1, value2), dim=1)

        return value.max(1)[0]

    def numpy_2D_Hausdorff_distance(self, x, y):
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        distance_matrix = np.linalg.norm(x[:, None] - y, axis=-1)

        value1 = np.max(np.min(distance_matrix, axis=2), axis=1, keepdims=True)
        value2 = np.max(np.min(distance_matrix, axis=1), axis=1, keepdims=True)

        value = np.concatenate((value1, value2), axis=1)

        return np.max(value, axis=1)

    def hausdorff_distance(self, contour1, contour2):
        # 对于contour1中的每个点，找到最近的点在contour2中的距离
        h1 = max([min([np.linalg.norm(a - b) for b in contour2]) for a in contour1])
        # 对于contour2中的每个点，找到最近的点在contour1中的距离
        h2 = max([min([np.linalg.norm(b - a) for a in contour1]) for b in contour2])

        return max(h1, h2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def my_los(self, gt_mask, pred_mask):
        deepcopied_tensor = gt_mask.clone().detach().cpu().numpy()

        deepcopied_tensor2 = pred_mask.clone().detach().cpu().numpy()
        all_loss = 0
        for index in range(len(deepcopied_tensor)):

            # a = (self.sigmoid(deepcopied_tensor[index]) * 255).astype(np.uint8)
            a = (deepcopied_tensor[index] * 255).astype(np.uint8)

            b = (self.sigmoid(deepcopied_tensor2[index]) * 255).astype(np.uint8)

            # 提取预测掩码的轮廓
            pred_contours, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_height, image_width = a.shape
            # normalized_pred_contours = []
            #
            # for contour in pred_contours:
            #     normalized_contour = []
            #     for point in contour:
            #         x, y = point[0]
            #         normalized_x = x / image_width  # 归一化 x 坐标
            #         normalized_y = y / image_height  # 归一化 y 坐标
            #         normalized_contour.append([[normalized_x, normalized_y]])
            #     normalized_pred_contours.append(normalized_contour)

            # 提取真实掩码的轮廓
            gt_contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # normalized_gt_contours = []
            #
            # for contour in gt_contours:
            #     normalized_contour = []
            #     for point in contour:
            #         x, y = point[0]
            #         normalized_x = x / image_width  # 归一化 x 坐标
            #         normalized_y = y / image_height  # 归一化 y 坐标
            #         normalized_contour.append([[normalized_x, normalized_y]])
            #     normalized_gt_contours.append(normalized_contour)

            # # 将二值图像转换为三通道彩色图像
            overlay_a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
            overlay_b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
            #
            # # 在彩色图像上绘制轮廓
            cv2.drawContours(overlay_a, pred_contours, -1, (0, 255, 0), 2)  # 使用绿色绘制预测轮廓
            cv2.drawContours(overlay_b, gt_contours, -1, (0, 255, 0), 2)  # 使用红色绘制真实轮廓

            # -----------构造距离提取算子--------------------
            if len(pred_contours) == len(gt_contours) and len(gt_contours) == 1:
                d1 = self.hausdorff_distance(gt_contours[0], pred_contours[0])
                # d1 = self.hausdorff_distance(np.array(normalized_gt_contours[0]), np.array(normalized_pred_contours[0]))
                all_loss += d1
            else:
                once_loss = abs(len(gt_contours) - len(gt_contours)) / len(gt_contours)
                all_loss += once_loss
            # -----------构造距离提取算子--------------------
            # d2=self.hausdorff_distance(gt_contours[0],pred_contours[0])
            # print('2 当前距离的度量结果值为->',d2)

            cv2.imshow('overlay_a', overlay_a)
            cv2.imshow('overlay_b', overlay_b)
            cv2.waitKey(0)

        # loss_seg = all_loss / xyxy.shape[0]
        # sigmod_seg=1 / (1 + np.exp(-loss_seg))
        # print('当前距离的度量结果值为->', loss_seg,all_loss)
        # with open('seg_loss.csv', mode='a', encoding='utf8')as f:
        #     f.write(str(loss_seg) + ',' + str(loss_seg) + ',\n')

    def calculate_normalized_moments_similarity(self, preds, targets):
        """
        Calculate similarity based on normalized central moments.

        Args:
        preds (numpy.ndarray): Predicted segmentation map.
        targets (numpy.ndarray): Ground truth segmentation map.

        Returns:
        float: The similarity score based on normalized central moments.
        """

        # Calculate Hu Moments, which are based on normalized central moments
        moments_preds = cv2.HuMoments(cv2.moments(preds)).flatten()
        moments_targets = cv2.HuMoments(cv2.moments(targets)).flatten()
        eps = 1e-10  # 一个小的正数，用于避免取对数时的数值稳定性问题
        moments_preds = -np.sign(moments_preds) * np.log10(np.abs(moments_preds) + eps)
        moments_targets = -np.sign(moments_targets) * np.log10(np.abs(moments_targets) + eps)

        # Calculate similarity (smaller values indicate higher similarity)
        similarity = np.sum((moments_preds - moments_targets) ** 2)

        return similarity

    def hu_moments_tensor(self, m):
        """
        Calculate Hu moments for tensors.

        Args:
        m (dict): Moments dictionary.

        Returns:
        torch.Tensor: Hu moments tensor.
        """
        # Calculate Hu moments
        hu = []
        hu.append(m['n20'] + m['n02'])
        hu.append((m['n20'] - m['n02']) ** 2 + 4 * m['n11'] ** 2)
        hu.append((m['n30'] - 3 * m['n12']) ** 2 + (3 * m['n21'] - m['n03']) ** 2)
        hu.append((m['n30'] + m['n12']) ** 2 + (m['n21'] + m['n03']) ** 2)
        hu.append((m['n30'] - 3 * m['n12']) * (m['n30'] + m['n12']) * (
                (m['n30'] + m['n12']) ** 2 - 3 * (m['n21'] + m['n03']) ** 2) + (3 * m['n21'] - m['n03']) * (
                          m['n21'] + m['n03']) * (3 * (m['n30'] + m['n12']) ** 2 - (m['n21'] + m['n03']) ** 2))
        hu.append((m['n20'] - m['n02']) * ((m['n30'] + m['n12']) ** 2 - (m['n21'] + m['n03']) ** 2) + 4 * m['n11'] * (
                m['n30'] + m['n12']) * (m['n21'] + m['n03']))
        hu.append((3 * m['n21'] - m['n03']) * (m['n30'] + m['n12']) * (
                (m['n30'] + m['n12']) ** 2 - 3 * (m['n21'] + m['n03']) ** 2) - (m['n30'] - 3 * m['n12']) * (
                          m['n21'] + m['n03']) * (3 * (m['n30'] + m['n12']) ** 2 - (m['n21'] + m['n03']) ** 2))

        return torch.stack(hu)

    def calculate_moments_tensor(self, img):
        """
        Calculate image moments for tensors.

        Args:
        img (torch.Tensor): Image tensor.

        Returns:
        dict: Moments dictionary.
        """
        assert len(img.shape) == 2, "Image should be grayscale"
        h, w = img.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        # Convert to float and calculate raw moments
        img = img.float()
        m00 = torch.sum(img)
        m10 = torch.sum(x * img)
        m01 = torch.sum(y * img)
        m20 = torch.sum(x ** 2 * img)
        m02 = torch.sum(y ** 2 * img)
        m11 = torch.sum(x * y * img)
        m30 = torch.sum(x ** 3 * img)
        m03 = torch.sum(y ** 3 * img)
        m12 = torch.sum(x * y ** 2 * img)
        m21 = torch.sum(x ** 2 * y * img)

        # Calculate central moments
        mu20 = m20 - m10 ** 2 / m00
        mu02 = m02 - m01 ** 2 / m00
        mu11 = m11 - m10 * m01 / m00
        mu30 = m30 - 3 * m21
        mu03 = m03 - 3 * m12
        mu12 = m12 - 2 * m21 - m10 * m01 ** 2 / m00
        mu21 = m21 - 2 * m12 - m01 * m10 ** 2 / m00

        # Normalize central moments
        mu20 /= m00 ** 2
        mu02 /= m00 ** 2
        mu11 /= m00 ** 2
        mu30 /= m00 ** 2.5
        mu03 /= m00 ** 2.5
        mu12 /= m00 ** 2.5
        mu21 /= m00 ** 2.5

        return {'n20': mu20, 'n02': mu02, 'n11': mu11, 'n30': mu30, 'n03': mu03, 'n12': mu12, 'n21': mu21}

    def calculate_contour_similarity(self, preds, targets):
        """
        Calculate the similarity of two images based on their Hu moments.

        Args:
        preds (torch.Tensor): Predicted image tensor.
        targets (torch.Tensor): Target image tensor.

        Returns:
        float: Similarity score.
        """
        moments_preds = self.calculate_moments_tensor(preds)
        moments_targets = self.calculate_moments_tensor(targets)

        hu_moments_preds = self.hu_moments_tensor(moments_preds)
        hu_moments_targets = self.hu_moments_tensor(moments_targets)

        # Log transform for scale invariance
        hu_moments_preds = -torch.sign(hu_moments_preds) * torch.log10(torch.abs(hu_moments_preds))
        hu_moments_targets = -torch.sign(hu_moments_targets) * torch.log10(torch.abs(hu_moments_targets))

        # Calculate similarity
        similarity = torch.sum((hu_moments_preds - hu_moments_targets) ** 2)
        # Check for NaN values
        if torch.isnan(similarity).any():
            return torch.tensor(10.0)

        return similarity

    def dice_loss(self, pred, target, epsilon=1e-6):
        pred = torch.sigmoid(pred)  # 确保预测值在 0 和 1 之间
        pred = pred > 0.5

        # 计算交集
        intersection = (pred * target).sum(dim=(1, 2))  # 假设批次维度是第一个维度
        # 计算预测和实际标签的和
        total = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        # 计算 Dice 系数
        dice = 2 * intersection / (total + epsilon)
        # 计算 Dice Loss
        dice_loss = 1 - dice
        # 返回批次的平均 Dice Loss
        return dice_loss.mean()

    def dice_loss2(self, pred, target, epsilon=1e-6):
        pred = torch.sigmoid(pred)  # 确保预测值在 0 和 1 之间
        # 计算交集
        intersection = (pred * target).sum(dim=(1, 2))  # 假设批次维度是第一个维度
        # 计算预测和实际标签的和
        total = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        # 计算 Dice 系数
        dice = 2 * intersection / (total + epsilon)
        # 计算 Dice Loss
        dice_loss = 1 - dice
        # 返回批次的平均 Dice Loss
        return dice_loss.mean()

    def single_mask_loss_GPU(self, gt_mask, pred, proto, xyxy, area, mask, img):
        """Mask loss for one image."""
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])
        mask = mask.unsqueeze(0).repeat(pred_mask.shape[0], 1, 1)
        # print(pred_mask.shape)  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, mask, reduction='none')
        # loss2 = self.dice_loss(pred_mask, gt_mask)
        loss3 = self.dice_loss2(pred_mask, mask)
        # # self.my_los(gt_mask,pred_mask)
        # single_mask = pred_mask[0].cpu().detach().numpy().astype(np.uint8) * 255
        # # print(single_mask.shape)

        # moments_loss = self.calculate_normalized_moments_similarity(single_mask, label_mask)
        # 遍历批量中的每个样本
        '''
        
        
            label_mask = gt_mask[0].cpu().detach().numpy().astype(np.uint8) * 255
        mask_mask = mask.cpu().detach().numpy() * 255
        img = img.cpu().detach().numpy()
        # 将图像数据范围从 [0, 1] 转换为 [0, 255]
        img_np = (img * 255).astype(np.uint8)

        # 调整通道顺序：从CHW到HWC
        img_np = np.transpose(img_np, (1, 2, 0))

        # 如果图像是RGB，转换为BGR以供OpenCV正确显示
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 使用OpenCV显示图像

        # cv2.imshow('single_mask',single_mask)
        cv2.imshow('label_mask', label_mask)
        cv2.imshow('mask_mask', mask_mask.astype(np.uint8)[0, :, :])
        cv2.imshow('img_np', img_np)
        cv2.waitKey(0)
        
         results = []
        x1, y1, x2, y2 = torch.chunk(xyxy[:, :, None], 4, 1)
        for i in range(pred_mask.shape[0]):
            single_mask = pred_mask[i][int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
            # print(single_mask.shape)
            label_mask = gt_mask[i][int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
            calculate_loss = self.calculate_contour_similarity(single_mask, label_mask)
            # 处理NaN值
            if torch.isnan(calculate_loss).any():
                # 将NaN值替换为合适的数值，例如0或者一个较大的数
                calculate_loss[torch.isnan(calculate_loss)] = torch.tensor(0.0)

            # 处理无穷大 (inf)
            if torch.isinf(calculate_loss).any():
                # 将无穷大值替换为合适的数值，例如一个较大的数或者一个最大有限值
                max_value = torch.finfo(calculate_loss.dtype).max
                calculate_loss[torch.isinf(calculate_loss)] = max_value

            # 处理None
            if calculate_loss is None:
                # 将None替换为合适的张量值，例如一个张量里全是0
                calculate_loss = torch.tensor(0.0, device=single_mask.device)
            results.append(calculate_loss)
        calculate_losses_tensor = torch.tensor(results)
        '''

        # return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean() \
        # return torch.log10(calculate_losses_tensor.float().mean() + 1)
        # return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean() + loss3
        return loss.mean() + loss3

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """Mask loss for one image."""

        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        # # self.my_los(gt_mask,pred_mask)
        # single_mask = pred_mask[0].cpu().detach().numpy().astype(np.uint8) * 255
        # # print(single_mask.shape)
        # label_mask = gt_mask[0].cpu().detach().numpy().astype(np.uint8) * 255
        # moments_loss = self.calculate_normalized_moments_similarity(single_mask, label_mask)
        # 遍历批量中的每个样本
        all_mask = []
        all_label = []
        x1, y1, x2, y2 = torch.chunk(xyxy[:, :, None], 4, 1)
        for i in range(pred_mask.shape[0]):
            # sigmoid_mask = torch.sigmoid(pred_mask[i])
            # print(pred_mask[i].mean())
            # 取出单个样本的掩码，并确保它在CPU上

            single_mask = pred_mask[i].cpu().detach().numpy()
            single_mask = (single_mask > 0.9).astype(np.uint8) * 255
            # single_mask = single_mask.astype(np.uint8) * 255
            single_mask = single_mask[int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]

            # print(single_mask.shape)
            label_mask = gt_mask[i].cpu().detach().numpy().astype(np.uint8) * 255
            label_mask = label_mask[int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
            #     all_mask.append(single_mask)
            #     all_label.append(label_mask)
            # cv2.imshow(f"single_mask{i}", single_mask)
            # cv2.imshow(f"label_mask{i}", label_mask)
            all_label.append(label_mask)
            all_mask.append(single_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 使用多进程池
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.calculate_normalized_moments_similarity, all_mask, all_label))

        mean_loss_np = torch.tensor(np.array(results)).to('cuda:0')
        # print(mean_loss_np)
        # cv2.destroyAllWindows()
        # return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean() + torch.log10(torch.tensor(moments_loss) + 1)
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean() + (mean_loss_np / area).mean()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                    idx = target_gt_idx[i][fg_mask[i]]
                    gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    area = xyxy2xywh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    pred_kpt = pred_kpts[i][fg_mask[i]]
                    kpt_mask = gt_kpt[..., 2] != 0
                    loss[1] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
                    # kpt_score loss
                    if pred_kpt.shape[-1] == 3:
                        loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose / batch_size  # pose gain
        loss[2] *= self.hyp.kobj / batch_size  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / 64
        loss_items = loss.detach()
        return loss, loss_items
