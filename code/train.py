import torch
from torch.utils.data import DataLoader
import numpy as np
import time, os
import pickle
from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network_s import create_h0_strategy
from evaluation import Evaluation
from tqdm import tqdm
from scipy.sparse import coo_matrix

import pickle
import os
import networkx as nx
from generate_walk import Node2Vec

torch.backends.cudnn.enabled = False

# parse settings
setting = Setting()
setting.parse()
dir_name = os.path.dirname(setting.log_file)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
setting.log_file = setting.log_file + '_' + timestring
log = open(setting.log_file, 'w')

# print(setting)

# log_string(log, 'log_file: ' + setting.log_file)
# log_string(log, 'user_file: ' + setting.trans_user_file)
# log_string(log, 'loc_temporal_file: ' + setting.trans_loc_file)
# log_string(log, 'loc_spatial_file: ' + setting.trans_loc_spatial_file)
# log_string(log, 'interact_file: ' + setting.trans_interact_file)

# log_string(log, str(setting.lambda_user))
# log_string(log, str(setting.lambda_loc))

# log_string(log, 'W in AXW: ' + str(setting.use_weight))
# log_string(log, 'GCN in user: ' + str(setting.use_graph_user))
# log_string(log, 'spatial graph: ' + str(setting.use_spatial_graph))

message = ''.join([f'{k}: {v}\n' for k, v in vars(setting).items()])
log_string(log, message)

# load dataset
poi_loader = PoiDataloader(
    setting.max_users, setting.min_checkins)  # 0， 5*20+1
poi_loader.read(setting.dataset_file)
# print('Active POI number: ', poi_loader.locations())  # 18737 106994
# print('Active User number: ', poi_loader.user_count())  # 32510 7768
# print('Total Checkins number: ', poi_loader.checkins_count())  # 1278274

log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))

# create flashback trainer
with open(setting.trans_loc_file, 'rb') as f:  # transition POI graph
    transition_graph = pickle.load(f)  # 在cpu上
# transition_graph = top_transition_graph(transition_graph)
transition_graph = coo_matrix(transition_graph)

if setting.use_spatial_graph:
    with open(setting.trans_loc_spatial_file, 'rb') as f:  # spatial POI graph
        spatial_graph = pickle.load(f)  # 在cpu上
    # spatial_graph = top_transition_graph(spatial_graph)
    spatial_graph = coo_matrix(spatial_graph)
else:
    spatial_graph = None

if setting.use_graph_user:
    with open(setting.trans_user_file, 'rb') as f:
        friend_graph = pickle.load(f)  # 在cpu上
    # friend_graph = top_transition_graph(friend_graph)
    friend_graph = coo_matrix(friend_graph)
else:
    friend_graph = None

with open(setting.trans_interact_file, 'rb') as f:  # User-POI interaction graph
    interact_graph = pickle.load(f)  # 在cpu上
interact_graph = csr_matrix(interact_graph)

log_string(log, 'Successfully load graph')

print("start to load node2vec!")
graph_poi = transition_graph
graph_poi = graph_poi.toarray()#[0]
G_graph_poi_forward = nx.from_numpy_array(graph_poi, create_using=nx.DiGraph)
walk_forward_func = Node2Vec(G_graph_poi_forward, workers=1, walk_length=3, num_walks=setting.sample)
print("successfully load node2vec!")



dataset = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TRAIN, G_graph_poi_forward, walk_forward_func, setting)  # 20, 200 or 1024, 0
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TEST, G_graph_poi_forward, walk_forward_func, setting)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(
), 'batch size must be lower than the amount of available users'
poi2gps = poi_loader.poi2gps

trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s, setting.lambda_loc, setting.lambda_user,
                           setting.use_weight, transition_graph, spatial_graph, friend_graph, setting.use_graph_user,
                           setting.use_spatial_graph, interact_graph, G_graph_poi_forward, setting)  # 0.01, 100 or 1000
h0_strategy = create_h0_strategy(
    setting.hidden_dim, setting.is_lstm)  # 10 True or False
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.mlp_hidden_dim, setting.rnn_factory,
                setting.device)
evaluation_test = Evaluation(dataset_test, dataloader_test,
                             poi_loader.user_count(), h0_strategy, trainer, setting, log)
print('{} {}'.format(trainer, setting.rnn_factory))

#  training loop
optimizer_t = torch.optim.Adam(trainer.parameters_t(
), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler_t = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_t, milestones=[30, 45, 55, 80], gamma=0.2)

optimizer_s = torch.optim.Adam(trainer.parameters_s(
), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler_s = torch.optim.lr_scheduler.MultiStepLR(
    optimizer_s, milestones=[30, 45, 55, 80], gamma=0.2)

param_count = trainer.count_parameters()
log_string(log, f'In total: {param_count} trainable parameters')

print("load parameters")
# trainer.save_parameters()
trainer.load_parameters_t()
trainer.load_parameters_s()
print("load parameters successful")
e = 1
evaluation_test.evaluate(G_graph_poi_forward, e, poi2gps)

bar_tea = tqdm(total=setting.tea_epochs)
bar_tea.set_description('Teacher Training')


for e in range(setting.tea_epochs):  # 100
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    dataset.shuffle_users()  # shuffle users before each epoch!

    losses = []
    epoch_start = time.time()
    for i, (x_real, x, x_adj, indexs, indexs_u, indexs_m, t, t_slot, s, s_real, y, y_t, y_t_slot, y_s, reset_h, active_users) in enumerate(dataloader):
        # reset hidden states for newly added users
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])

        indexs = indexs.squeeze(0).to(setting.device)[0]
        seq_len = torch.max(indexs) + 1
        x = x.squeeze(0)[0, :seq_len].to(setting.device)
        indexs_u = indexs_u.squeeze(0)[0, :seq_len].to(setting.device)
        indexs_m = indexs_m.squeeze(0)[0, :seq_len].to(setting.device)
        x_real = x_real.squeeze(0).to(setting.device)
        x_adj = x_adj.squeeze(0).to(setting.device)


        t = t.squeeze(0).to(setting.device)
        t_slot = t_slot.squeeze(0).to(setting.device)
        
        s = s.squeeze(0)[:seq_len].to(setting.device)
        s_real = s_real.squeeze(0).to(setting.device)

        y = y.squeeze(0).to(setting.device)
        y_t = y_t.squeeze(0).to(setting.device)
        y_t_slot = y_t_slot.squeeze(0).to(setting.device)
        y_s = y_s.squeeze(0).to(setting.device)
        active_users = active_users.to(setting.device)

        optimizer_s.zero_grad()
        loss = trainer.loss_s(x_real, x, x_adj, indexs, indexs_u, indexs_m, t, t_slot, s, s_real, y, y_t,
                            y_t_slot, y_s, h, active_users)

        loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(trainer.parameters(), 5)
        losses.append(loss.item())
        optimizer_s.step()

    # schedule learning rate:
    scheduler_s.step()
    bar_tea.update(1)
    epoch_end = time.time()
    log_string(log, 'One teacher training need {:.2f}s'.format(
        epoch_end - epoch_start))
    
    # statistics:
    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        log_string(log, f'Teacher Epoch: {e + 1}/{setting.tea_epochs}')
        log_string(log, f'Used learning rate: {scheduler_s.get_last_lr()[0]}')
        log_string(log, f'Avg Loss: {epoch_loss}')

    if (e + 1) % 35 == 0:   #setting.validate_epoch
        log_string(log, f'~~~ Test Teacher Set Evaluation (Epoch: {e + 1}) ~~~')
        evl_start = time.time()
        evaluation_test.evaluate(G_graph_poi_forward, e, poi2gps)
        evl_end = time.time()
        log_string(log, 'One teacher evaluate need {:.2f}s'.format(
            evl_end - evl_start))

bar_tea.close()

print("load parameters")
trainer.save_parameters_t()
trainer.load_parameters_s()
print("load parameters successful")

bar = tqdm(total=setting.epochs)
bar.set_description('Student Training')

for e in range(setting.epochs):  # 100
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    dataset.shuffle_users()  # shuffle users before each epoch!

    losses = []
    stu_losses = []
    tea_losses = []
    kd_losses = []
    weight_losses = []
    inter_losses = []
    epoch_start = time.time()
    for i, (x_real, x, x_adj, indexs, indexs_u, indexs_m, t, t_slot, s, s_real, y, y_t, y_t_slot, y_s, reset_h, active_users) in enumerate(dataloader):
        # reset hidden states for newly added users
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])

        indexs = indexs.squeeze(0).to(setting.device)[0]
        seq_len = torch.max(indexs) + 1
        x = x.squeeze(0)[0, :seq_len].to(setting.device)
        indexs_u = indexs_u.squeeze(0)[0, :seq_len].to(setting.device)
        indexs_m = indexs_m.squeeze(0)[0, :seq_len].to(setting.device)
        x_real = x_real.squeeze(0).to(setting.device)
        x_adj = x_adj.squeeze(0).to(setting.device)


        t = t.squeeze(0).to(setting.device)
        t_slot = t_slot.squeeze(0).to(setting.device)
        
        s = s.squeeze(0)[:seq_len].to(setting.device)
        s_real = s_real.squeeze(0).to(setting.device)

        y = y.squeeze(0).to(setting.device)
        y_t = y_t.squeeze(0).to(setting.device)
        y_t_slot = y_t_slot.squeeze(0).to(setting.device)
        y_s = y_s.squeeze(0).to(setting.device)
        active_users = active_users.to(setting.device)

        optimizer_s.zero_grad()
        loss, stu_loss, kd_loss, tea_loss, weight_loss, inter_loss = trainer.loss_s(x_real, x, x_adj, indexs, indexs_u, indexs_m, t, t_slot, s, s_real, y, y_t,
                            y_t_slot, y_s, h, active_users, e)

        loss.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(trainer.parameters_s(), 5)
        losses.append(loss.item())
        stu_losses.append(stu_loss.item())
        tea_losses.append(tea_loss.item())
        kd_losses.append(kd_loss.item())
        weight_losses.append(weight_loss.item())
        inter_losses.append(inter_loss.item())
        optimizer_s.step()

    # schedule learning rate:
    scheduler_s.step()
    bar.update(1)
    epoch_end = time.time()
    log_string(log, 'One student training need {:.2f}s'.format(
        epoch_end - epoch_start))
    # statistics:
    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        epoch_stu_loss = np.mean(stu_losses)
        epoch_tea_loss = np.mean(tea_losses)
        epoch_kd_loss = np.mean(kd_losses)
        epoch_weight_loss = np.mean(weight_losses)
        epoch_inter_loss = np.mean(inter_losses)
        print('Student Epoch: ', f'{e + 1}/{setting.epochs}', 'Used learning rate: ', scheduler_s.get_last_lr()[0], 'Avg Loss: ', epoch_loss, 'Student Loss: ', epoch_stu_loss, 'Teacher Loss: ', epoch_tea_loss, 'KD Loss: ', epoch_kd_loss, 'weight Loss: ', epoch_weight_loss, 'inter Loss: ', epoch_inter_loss)
        # log_string(log, f'Student Epoch: {e + 1}/{setting.epochs}')
        # log_string(log, f'Used learning rate: {scheduler_s.get_last_lr()[0]}')
        # log_string(log, f'Avg Loss: {epoch_loss}')
        # log_string(log, f'Student Loss: {epoch_stu_loss}')
        # log_string(log, f'Teacher Loss: {epoch_tea_loss}')
        # log_string(log, f'KD Loss: {epoch_kd_loss}')



    if (e + 1) % 50 == 0:  #setting.validate_epoch
        log_string(log, f'~~~ Test Student Set Evaluation (Epoch: {e + 1}) ~~~')
        evl_start = time.time()
        evaluation_test.evaluate(G_graph_poi_forward, e, poi2gps)
        evl_end = time.time()
        log_string(log, 'One evaluate need {:.2f}s'.format(
            evl_end - evl_start))

bar.close()

# print("load parameters")
# trainer.save_parameters()
# #trainer.load_parameters_t()
# print("load parameters successful")