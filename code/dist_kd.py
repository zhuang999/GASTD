import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    y_s, y_t = y_s.transpose(1, 2), y_t.transpose(1, 2)
    a, b = y_s - y_s.mean(2).unsqueeze(2), y_t - y_t.mean(2).unsqueeze(2)
    cosine_value = (a * b).sum(2) / (a.norm(dim=2) * b.norm(dim=2) + 1e-8)
    inter_relation = 1 - cosine_value.mean()
    return inter_relation


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t, loc_count, attention):
        y_s = (z_s / self.tau).softmax(dim=2)
        #y_s = F.log_softmax(z_s / self.tau, dim=1)
        y_t = (z_t / self.tau).softmax(dim=2)

        inter_loss = inter_class_relation(y_s.view(-1, loc_count), y_t.view(-1, loc_count))
        intra_loss = intra_class_relation(y_s, y_t)   #self.tau**2
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss