import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from torch.nn.parameter import Parameter
from torch.autograd import Variable

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class EMA():
    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.cpu().detach()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.decay) * x.cpu().detach() + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()
def pairwise_distance(A, B):
    """
    Compute distance between points in A and points in B
    :param A:  (m,n) -m points, each of n dimension. Every row vector is a point, denoted as A(i).
    :param B:  (k,n) -k points, each of n dimension. Every row vector is a point, denoted as B(j).
    :return:  Matrix with (m, k). And the ele in (i,j) is the distance between A(i) and B(j)
    """
    A_square = torch.sum(A * A, dim=1, keepdim=True)
    B_square = torch.sum(B * B, dim=1, keepdim=True)

    distance = A_square + B_square.t() - 2 * torch.matmul(A, B.t())

    return distance

def constraints(features, labels):
    labels = torch.reshape(labels, (labels.shape[0], 1))
    con_loss = AverageMeter()
    index_dict = {k.item() for k in labels}
    for index in index_dict:
        labels_mask = (labels == index)
        feas = torch.masked_select(features, labels_mask)
        feas = feas.view(-1, features.shape[1])
        distance = pairwise_distance(feas, feas)
        # torch.sqrt_(distance)
        num = feas.shape[0] * (feas.shape[0] - 1)
        loss = torch.sum(distance) / num
        con_loss.update(loss, n=num / 2)
    return con_loss.avg


def constraints_loss(data_loader, network, args):
    network.eval()
    max_size = args.batch_size * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size, args.feature_size)).cuda()
    audio_bank = torch.zeros((max_size, args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    index = 0
    con_images = 0.0
    con_text = 0.0
    con_audio = 0.0
    with torch.no_grad():
        for captions, audio, images, labels in data_loader:
            images = images.cuda()
            captions = captions.cuda()
            audio = audio.cuda()
            labels = labels.cuda()
            interval = images.shape[0]
            text_embeddings, audio_embeddings, image_embeddings = network(captions, audio, images, labels)

            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            audio_bank[index: index + interval] = audio_embeddings

            labels_bank[index: index + interval] = labels
            index = index + interval
        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        audio_bank = audio_bank[:index]
        labels_bank = labels_bank[:index]

    if args.constraints_text:
        con_text = constraints(text_bank, labels_bank)
    if args.constraints_images:
        con_images = constraints(images_bank, labels_bank)
    if args.constraints_audios:
        con_audio = constraints(audio_bank, labels_bank)

    return con_images, con_text, con_audio
def gcs_divergence(pred_1, pred_2, pred_3, target, epsilon=1e-8):
    """
    Compute the CS divergence.
    :param pred: Predicted distribution (P).
    :param target: Target distribution (Q).
    :param epsilon: Small value for numerical stability.
    :return: CS divergence value.
    """
    # Ensure numerical stability
    pred_1 = torch.clamp(pred_1, min=epsilon)
    pred_2 = torch.clamp(pred_2, min=epsilon)
    pred_3 = torch.clamp(pred_3, min=epsilon)
    target = torch.clamp(target, min=epsilon)

    # Compute the numerator and denominator of the CS divergence formula
    numerator = torch.sum(pred_1 * pred_2 *pred_3 * target, dim=1)
    denominator = (torch.sum(pred_1 ** 4, dim=1) ** (1/4)) * (torch.sum(pred_2 ** 4, dim=1) ** (1/4)) * (torch.sum(pred_3 ** 4, dim=1) ** (1/4)) * (torch.sum(target ** 4, dim=1) ** (1/4))
    # Compute the CS divergence
    cs_loss = -torch.log(numerator / denominator)
    cs_loss = torch.mean(cs_loss)
    return cs_loss

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.epsilon = 1e-8
        self.num_classes = args.num_classes
        self.tau = 0.1
        self.lambda_recon = 0.1
        if args.resume:
            checkpoint = torch.load(args.model_path)
            self.W = nn.Parameter(checkpoint['W'])
            print('=========> Loading in parameter W from pretrained models')
        else:
            self.W = nn.Parameter(torch.randn(args.feature_size, args.num_classes))
            self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)


    def compute_similarity_score(self, e_t_i, e_m_i, temperature=0.05):
        """
        计算单个文本-音频对的相似度得分
        """
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(e_t_i.unsqueeze(0), e_m_i, dim=1)
        
        # 应用温度参数
        scaled_sim = torch.exp(cos_sim / temperature)
        # 计算 softmax 分母
        sum_exp_sim = torch.sum(torch.exp(scaled_sim), dim=0, keepdim=True)
        
        # 计算 softmax 相似度得分
        softmax_similarity_score = torch.exp(scaled_sim) / sum_exp_sim
        return softmax_similarity_score

    def compute_similarity_matrix(self, query_feats, key_feats, temperature=0.05):
        """
        优化后的向量化相似度矩阵计算
        query_feats: (B, 1, D)
        key_feats: (B, 1, D)
        返回: (B, B) 相似度矩阵
        """
        # 移除多余的维度，确保是二维矩阵
        query_norm = F.normalize(query_feats.squeeze(1), p=2, dim=1)  # (B, D)
        key_norm = F.normalize(key_feats.squeeze(1), p=2, dim=1)  # (B, D)
        
        # 余弦相似度矩阵
        sim_matrix = torch.mm(query_norm, key_norm.transpose(0, 1))  # (B, B)
        
        # 应用温度缩放
        sim_matrix = torch.exp(sim_matrix / temperature)
        
        # 行方向归一化
        row_sums = sim_matrix.sum(dim=1, keepdim=True)
        return sim_matrix / (row_sums + self.epsilon)
    
    def create_target_matrix(self, labels):
        """
        基于标签生成目标矩阵
        labels: (B,)
        返回: (B, B) 目标相似度矩阵
        """
        B = labels.size(0)
        targets = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B,B)
        return targets.float()

    def compute_recall_at_k(self, similarity_scores, labels, k=10):
        """
        计算召回率。
        :param similarity_scores: 相似度得分矩阵，形状为 (batch_size, num_samples)。
        :param labels: 真实标签，形状为 (batch_size,)。
        :param k: 考虑的前 K 个检索结果。
        :return: 召回率。
        """
        batch_size = labels.size(0)
        recall = 0.0
        
        # 为每个查询计算召回率
        for i in range(batch_size):
            # 获取前 K 个最相似的样本索引
            _, top_k_indices = torch.topk(similarity_scores[i], k, largest=True)
            
            # 检查这些样本中是否有与查询样本相同类别的样本
            for idx in top_k_indices:
                if labels[i] == labels[idx]:
                    recall += 1
                    break  # 找到后即跳出循环
                    
        # 计算平均召回率
        recall /= batch_size
        return recall

    def reconstruction_loss(self, original, reconstructed):
        """
        计算重建损失，使用均方误差（MSE）。

        参数:
        original (torch.Tensor): 原始特征，形状为 (B, D)。
        reconstructed (torch.Tensor): 重建的特征，形状为 (B, D)。

        返回:
        torch.Tensor: 重建损失。
        """
        return F.mse_loss(original, reconstructed)



    def compute_cmcl_loss(self, image_embeddings, text_embeddings, audio_embeddings, labels):
        """
        计算跨模态对比学习损失。
        :param image_embeddings: 图像嵌入向量，形状为 (batch_size, latent_dim)。
        :param text_embeddings: 文本嵌入向量，形状为 (batch_size, latent_dim)。
        :param audio_embeddings: 音频嵌入向量，形状为 (batch_size, latent_dim)。
        :param labels: 真实标签，形状为 (batch_size,)。
        """
        # 归一化特征
        # 在 compute_cmcl_loss 函数中
        image_norm = F.normalize(image_embeddings, dim=1)  # (B, D)
        text_norm = F.normalize(text_embeddings, dim=1)  # (B, D)
        audio_norm = F.normalize(audio_embeddings, dim=1)  # (B, D)
        #print('image_embeddings',image_embeddings.shape)
        #print('text_embeddings',text_embeddings.shape)
        #print('text_embeddings',text_embeddings.shape)

        # 生成目标矩阵
        S_target = self.create_target_matrix(labels)
        
        # 计算各模态间的相似度矩阵
        S_t2a_pred = self.compute_similarity_matrix(text_norm, audio_norm, self.tau)
        S_a2t_pred = self.compute_similarity_matrix(audio_norm, text_norm, self.tau)
        S_t2i_pred = self.compute_similarity_matrix(text_norm, image_norm, self.tau)
        S_i2t_pred = self.compute_similarity_matrix(image_norm, text_norm, self.tau)
        S_a2i_pred = self.compute_similarity_matrix(audio_norm, image_norm, self.tau)
        S_i2a_pred = self.compute_similarity_matrix(image_norm, audio_norm, self.tau)
        

        
        # 计算GCS散度
        loss = 0
        loss = gcs_divergence(S_t2a_pred, S_t2i_pred, S_a2i_pred, S_target) + gcs_divergence(S_a2t_pred, S_i2t_pred, S_i2a_pred, S_target)

        
        # 计算召回率
        recall_t2a = self.compute_recall_at_k(S_t2a_pred, labels)
        recall_t2i = self.compute_recall_at_k(S_t2i_pred, labels)
        recall_a2i = self.compute_recall_at_k(S_a2i_pred, labels)

        return loss, recall_t2a, recall_t2i, recall_a2i

    def forward(self, image_embeddings, text_embeddings, audio_embeddings, reconstructed_audio_embeddings, labels):
        loss_cmcl, recall_t2a, recall_t2i, recall_a2i = self.compute_cmcl_loss(image_embeddings, text_embeddings, audio_embeddings, labels)
        loss_re = self.reconstruction_loss(audio_embeddings, reconstructed_audio_embeddings)
        loss = loss_cmcl + loss_re
        return loss, recall_t2a, recall_t2i, recall_a2i

class AverageMeter(object):
    """
    Computes and stores the averate and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count

def compute_topk(query, gallery, target_query, target_gallery, k=[1,10], reverse=False):
    result = []
    query = query / query.norm(dim=1,keepdim=True)
    gallery = gallery / gallery.norm(dim=1,keepdim=True)
    sim_cosine = torch.matmul(query, gallery.t())
    result.extend(topk(sim_cosine, target_gallery, target_query, k=[1,10]))
    if reverse:
        result.extend(topk(sim_cosine, target_query, target_gallery, k=[1,10], dim=0))
    return result


def topk(sim, target_gallery, target_query, k=[1,10], dim=1):
    result = []
    maxk = max(k)
    size_total = len(target_gallery)
    _, pred_index = sim.topk(maxk, dim, True, True)
    pred_labels = target_gallery[pred_index]
    if dim == 1:
        pred_labels = pred_labels.t()
    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))

    for topk in k:
        #correct_k = torch.sum(correct[:topk]).float()
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        result.append(correct_k * 100 / size_total)
    return result