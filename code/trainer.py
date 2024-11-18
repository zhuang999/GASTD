import time

import torch
import torch.nn as nn
import numpy as np
from utils import *
from network_t import Flashback
from network_s import MLPMixer
from dist_kd import DIST
from scipy.sparse import csr_matrix
from scipy.stats import entropy

class FlashbackTrainer():
    """ Instantiates Flashback module with spatial and temporal weight functions.
    Performs loss computation and prediction.
    """

    def __init__(self, lambda_t, lambda_s, lambda_loc, lambda_user, use_weight, transition_graph, spatial_graph,
                 friend_graph, use_graph_user, use_spatial_graph, interact_graph, graph_nx, args):
        """ The hyper parameters to control spatial and temporal decay.
        """
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph
        self.graph = transition_graph
        self.spatial_graph = spatial_graph
        self.friend_graph = friend_graph
        self.interact_graph = interact_graph
        self.graph_nx = graph_nx
        self.args = args

    def __str__(self):
        return 'Use flashback training.'

    def count_parameters(self):
        param_count = 0
        for name, param in self.model_s.named_parameters():
            if param.requires_grad:
                param_count += param.numel()
        return param_count
    
    def parameters_s(self):
        return self.model_s.parameters()
    
    def parameters_t(self):
        return self.model_t.parameters()

    def save_parameters_s(self):
        torch.save(self.model_s.state_dict(), "4sq_student.pt")
    
    def save_parameters_t(self):
        torch.save(self.model_t.state_dict(), "4sq_teacher_attn.pt")
    
    def load_parameters_t(self):
        self.model_t.load_state_dict(torch.load("gowalla_teacher_attn_new.pt", map_location={'cuda:0':'cuda:1'}), strict=False) #, map_location={'cuda:0':'cuda:1'}
    
    def load_parameters_s(self):
        self.model_s.load_state_dict(torch.load("gowalla_student_2.pt", map_location={'cuda:0':'cuda:1'}), strict=False) #, map_location={'cuda:0':'cuda:1'}

    def prepare(self, loc_count, user_count, hidden_size, mlp_hidden_size, gru_factory, device):
        def f_t(delta_t, user_len): return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t))  # hover cosine + exp decay

        # exp decay  2个functions
        def f_s(delta_s, user_len): return torch.exp(-(delta_s * self.lambda_s))
        self.loc_count = loc_count + 2
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cross_entropy_all_loss = nn.CrossEntropyLoss(reduction="none")
        self.criterion_s = nn.KLDivLoss(reduction="batchmean")  #, log_target=True
        self.criterion_s_all = nn.KLDivLoss(reduction="none")
        self.dist = DIST()
        self.model_t = Flashback(self.loc_count, user_count, hidden_size, f_t, f_s, gru_factory, self.lambda_loc,
                               self.lambda_user, self.use_weight, self.graph, self.spatial_graph, self.friend_graph,
                               self.use_graph_user, self.use_spatial_graph, self.interact_graph, self.graph_nx, self.args).to(device)
        self.model_s = MLPMixer(self.loc_count, user_count, mlp_hidden_size, f_t, f_s, gru_factory, self.lambda_loc,
                               self.lambda_user, self.use_weight, self.graph, self.spatial_graph, self.friend_graph,
                               self.use_graph_user, self.use_spatial_graph, self.interact_graph, self.graph_nx, self.args).to(device)
    
    def evaluate_t(self, x_real, x, x_adj, indexs, t, t_slot, s, s_real, y_t, y_t_slot, y_s, h, active_users):
        """ takes a batch (users x location sequence)
        then does the prediction and returns a list of user x sequence x location
        describing the probabilities for each location at each position in the sequence.
        t, s are temporal and spatial data related to the location sequence x
        y_t, y_s are temporal and spatial data related to the target sequence y.
        Flashback does not access y_t and y_s for prediction! 
        """
        self.model_t.eval()
        out_t, h, t_distance = self.model_t(x_real, x, x_adj, indexs, t, t_slot, s_real, s, y_t,
                            y_t_slot, y_s, h, active_users)
        out_t = out_t.transpose(0, 1)
        return out_t, h  # model outputs logits

    def evaluate_s(self, x_real, x, x_adj, indexs, t, t_slot, s, s_real, y_t, y_t_slot, y_s, h, active_users):
        """ takes a batch (users x location sequence)
        then does the prediction and returns a list of user x sequence x location
        describing the probabilities for each location at each position in the sequence.
        t, s are temporal and spatial data related to the location sequence x
        y_t, y_s are temporal and spatial data related to the target sequence y.
        Flashback does not access y_t and y_s for prediction! 
        """

        self.model_s.eval()
        # (seq_len, user_len, loc_count)
        out_s, h, s_distance = self.model_s(x_real, x, x_adj, indexs, t, t_slot, s_real, s, y_t,
                            y_t_slot, y_s, h, active_users)
        

        out_s = out_s.transpose(0, 1)

        return out_s, h  # model outputs logits
    

    def loss_t(self, x_real, x, x_adj, indexs, indexs_u, indexs_m, t, t_slot, s, s_real, y, y_t, y_t_slot, y_s, h, active_users):
        """ takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss """

        self.model_t.train()
        
        output_t, h, _ = self.model_t(x_real, x, x_adj, indexs, t, t_slot, s_real, s, y_t, y_t_slot, y_s, h,
                            active_users)  # out (seq_len, batch_size, loc_count)
        out_t = output_t.view(-1, self.loc_count)  # (seq_len * batch_size, loc_count)
        y = y.view(-1)  # (seq_len * batch_size)
        tea_loss = self.cross_entropy_loss(out_t, y) 
        

        return tea_loss
    
    def loss_s(self, x_real, x, x_adj, indexs, indexs_u, indexs_m, t, t_slot, s, s_real, y, y_t, y_t_slot, y_s, h, active_users):
        """ takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss """

        self.model_t.train()
        seq_len, batch_size = y.shape[0], y.shape[1]
        output_t, h, kd_weight = self.model_t(x_real, x, x_adj, indexs, t, t_slot, s_real, s, y_t, y_t_slot, y_s, h,
                            active_users)  # out (seq_len, batch_size, loc_count)
        out_t = output_t.view(-1, self.loc_count)  # (seq_len * batch_size, loc_count)


        output_s, h, s_distance = self.model_s(x_real, x, x_adj, indexs, t, t_slot, s_real, s, y_t, y_t_slot, y_s, h,
                            active_users)  # out (seq_len, batch_size, loc_count)
        out_s = output_s.view(-1, self.loc_count)  # (seq_len * batch_size, loc_count)
        y_shape = y.view(-1)  # (seq_len * batch_size)

        tea_all_loss = entropy(out_t)
        tea_index = torch.argmax(out_t, dim=-1)

        
        tea_index = torch.argsort(out_t, axis=-1)[:, -10:]
        y_repeated = y_shape.unsqueeze(1).repeat(1, 10)
        mask_value_tea = (tea_index == y_repeated) #* torch.exp(torch.arange(0, 1, 0.1)).unsqueeze(0).to(y.device) #F.softmax(out_t[tea_index], dim=-1)
        mask_value_tea_pos = mask_value_tea.sum(-1)
        mask_value_tea_neg = 1 - mask_value_tea_pos

        stu_index = torch.argsort(out_s, axis=-1)[:, -10:]
        mask_value_stu = (stu_index == y_repeated).unsqueeze(0).to(y.device) #F.softmax(out_t[tea_index], dim=-1)
        mask_value_stu_pos = mask_value_stu.sum(-1)
        mask_value_stu_neg = 1 - mask_value_stu.sum(-1)

        mask_value_st_np = mask_value_tea_pos * mask_value_stu_neg
        

        attention_p = mask_value_st_np # torch.exp(-tea_all_loss_e) * mask_value_st_np


        temp = 0.5
        stu_loss = self.cross_entropy_loss(out_s, y_shape)
        

        weight_loss = self.criterion_s_all(torch.log(s_distance.permute(2,0,1).reshape(-1, seq_len)),
            kd_weight.permute(2,0,1).reshape(-1, seq_len)).sum(dim=-1)#.sum()/ y.shape[0]
        weight_loss = torch.mul(attention_p, weight_loss).mean()#.sum() / weight_loss.shape[0]


        inter_loss = self.criterion_s(F.log_softmax(output_s.permute(1,2,0).reshape(-1, seq_len), dim=-1),
            F.softmax(output_t.permute(1,2,0).reshape(-1, seq_len),dim=-1))

        alpha = 0.5
        loss = alpha * stu_loss + (1 - alpha) * (weight_loss + inter_loss) #* (temp**2) #if (tea_loss - stu_loss < 0) else stu_loss

        return loss#, stu_loss, kd_loss, tea_loss, weight_loss, inter_loss


#kl_loss: https://blog.csdn.net/ChenglinBen/article/details/122057981

#weight_loss加上attention效果更好