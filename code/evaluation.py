import torch
import numpy as np
from utils import log_string, haversine
import torch.nn.functional as F

class Evaluation:
    """
    Handles evaluation on a given POI dataset and loader.

    The two metrics are MAP and recall@n. Our model predicts sequence of
    next locations determined by the sequence_length at one pass. During evaluation we
    treat each entry of the sequence as single prediction. One such prediction
    is the ranked list of all available locations and we can compute the two metrics.

    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.

    Using the --report_user argument one can access the statistics per user.
    """

    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting, log):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting
        self._log = log

    def evaluate(self, graph, epoch, poi2gps):
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)

        with torch.no_grad():
            iter_cnt = 0
            recall1 = 0
            recall5 = 0
            recall10 = 0
            average_precision = 0.

            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)
            reset_count = torch.zeros(self.user_count)

            right_single_distance = {}
            error_single_distance = {}
            adj_single_distance = {}
            right_average_distance = []
            right_average_time = []
            error_average_distance = []
            error_average_time = []
            error_predict_average_distance = []
            adj_average_distance = []

            all_distance = []
            all_time = []
            error_set = {}
            confidence_s_all,confidence_t_all,entropy_s_all,entropy_t_all,distance_all, distance_t_all, distance_s_all = [],[],[],[],[],[],[]
            false_negative_distance = []
            false_negative_entropy = []
            
            for i, (x_real, x, x_adj, index, index_u, index_m, t, t_slot, s, s_real, y, y_t, y_t_slot, y_s, reset_h, active_users) in enumerate(self.dataloader):
                active_users = active_users.squeeze()
                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        reset_count[active_users[j]] += 1

                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze(0).to(self.setting.device)[0]
                x_real = x_real.squeeze(0).to(self.setting.device)
                index = index.squeeze(0).to(self.setting.device)[0]
                times = t.squeeze(0).to(self.setting.device)
                t_slot = t_slot.squeeze(0).to(self.setting.device)
                s = s.squeeze(0).to(self.setting.device)
                s_real = s_real.squeeze(0).to(self.setting.device)
                x_adj = x_adj.squeeze(0).to(self.setting.device)

                y = y.squeeze(0)
                y_t = y_t.squeeze(0).to(self.setting.device)
                y_t_slot = y_t_slot.squeeze(0).to(self.setting.device)
                y_s = y_s.squeeze(0).to(self.setting.device)
                active_users = active_users.to(self.setting.device)

                # evaluate:
                out_t, _ = self.trainer.evaluate_t(x_real, x, x_adj, index, times, t_slot, s, s_real, y_t, y_t_slot, y_s, h, active_users)
                out_s, _ = self.trainer.evaluate_s(x_real, x, x_adj, index, times, t_slot, s, s_real, y_t, y_t_slot, y_s, h, active_users)

                for j in range(self.setting.batch_size):
                    # o contains a per user list of votes for all locations for each sequence entry
                    o_t = out_t[j]
                    o_s = out_s[j]

                    # partition elements
                    o_n_t = o_t.cpu().detach().numpy()
                    o_n_s = o_s.cpu().detach().numpy()
                    ind_t = np.argpartition(o_n_t, -10, axis=1)[:, -10:]  # top 10 elements
                    ind_s = np.argpartition(o_n_s, -10, axis=1)[:, -10:]  # top 10 elements

                    # o_n_t = o_t.cpu().detach().numpy()
                    # ind_t = np.argpartition(o_n_t, -10, axis=1)[:, -10:]  # top 10 elements

                    y_j = y[:, j]
                    x_j = x_real[:, j]
                    y_t_j = y_t[:, j].cpu()
                    t_j = times[:, j].cpu()


                    for k in range(len(y_j)):
                        if reset_count[active_users[j]] > 1:
                            continue  # skip already evaluated users.

                        # resort indices for k:
                        ind_k_t = ind_t[k]
                        r_t = ind_k_t[np.argsort(-o_n_t[k, ind_k_t], axis=0)]  # sort top 10 elements descending
                        ind_k_s = ind_s[k]
                        r_s = ind_k_s[np.argsort(-o_n_s[k, ind_k_s], axis=0)]  # sort top 10 elements descending

                        # ind_k_t = ind_t[k]
                        # r_t = ind_k_t[np.argsort(-o_n_t[k, ind_k_t], axis=0)]  # sort top 10 elements descending
                        # r_t = torch.tensor(r_t)

                        r_t = torch.tensor(r_t)
                        r_s = torch.tensor(r_s)
                        t = y_j[k]
                        s = x_j[k]
                        tt = y_t_j[k]
                        ts = t_j[k]

                        # compute MAP:
                        r_kj_t = o_n_t[k, :]
                        t_val_t = r_kj_t[t]
                        upper_t = np.where(r_kj_t > t_val_t)[0]
                        precision_t = 1. / (1 + len(upper_t))
                        #precision = precision_t
                        r_kj_s = o_n_s[k, :]
                        t_val_s = r_kj_s[t]
                        upper_s = np.where(r_kj_s > t_val_s)[0]
                        precision_s = 1. / (1 + len(upper_s))
                        precision = precision_t if precision_t >= precision_s else precision_s
                        # store
                        u_iter_cnt[active_users[j]] += 1
                        u_recall1[active_users[j]] += t in r_t[:1]
                        u_recall5[active_users[j]] += t in r_t[:5]
                        u_recall10[active_users[j]] += t in r_t[:10]
                        u_average_precision[active_users[j]] += precision

                        if t not in r_t[:1]:
                            u_recall1[active_users[j]] += t in r_s[:1]
                        if t not in r_t[:5]:
                            u_recall5[active_users[j]] += t in r_s[:5]
                        if t not in r_t[:10]:
                            u_recall10[active_users[j]] += t in r_s[:10]

                        #plot the distribution
                        t, s = t.item(), s.item()
                        distance = haversine(poi2gps[t][0], poi2gps[t][1], poi2gps[s][0], poi2gps[s][1])
                        distance_t = haversine(poi2gps[t][0], poi2gps[t][1], poi2gps[r_t[0].item()][0], poi2gps[r_t[0].item()][1])
                        distance_s = haversine(poi2gps[t][0], poi2gps[t][1], poi2gps[r_s[0].item()][0], poi2gps[r_s[0].item()][1])
                        prob_t = F.softmax(o_t[k],dim=-1)
                        log_prob_t = F.log_softmax(o_t[k], dim=-1)
                        entropy_t = -torch.sum(prob_t * log_prob_t, dim=-1)
                        confidence_t = 1- (entropy_t / torch.log(torch.tensor(o_t.size(-1))))
                        prob_s = F.softmax(o_s[k],dim=-1)
                        log_prob_s = F.log_softmax(o_s[k], dim=-1)
                        entropy_s = -torch.sum(prob_s * log_prob_s, dim=-1)
                        confidence_s = 1- (entropy_s / torch.log(torch.tensor(o_s.size(-1))))
                        distance_all.append(distance)
                        distance_t_all.append(distance_t)
                        distance_s_all.append(distance_s)
                        # entropy_t_all.append(entropy_t.item())
                        # entropy_s_all.append(entropy_s.item())
                        # confidence_t_all.append(confidence_t.item())
                        # confidence_s_all.append(confidence_s.item())
                        if t in r_t[:1] and t not in r_s[:1]:
                            false_negative_distance.append(distance_t)
                            false_negative_entropy.append(entropy_t.item())
                        # if t in r_t[:10] and t in r_s[:10]:
                        #     confidence_t_all.append(confidence_t.item())
                        #     confidence_s_all.append(confidence_s.item())
                        if t in r_t[:1] and t in r_s[:1]:
                            entropy_t_all.append(entropy_t.item())
                            entropy_s_all.append(entropy_s.item())
                            confidence_t_all.append(confidence_t.item())
                            confidence_s_all.append(confidence_s.item())

                        # dict = {'confidence_s':confidence_s_all, 'confidence_t':confidence_t_all, 'entropy_s':entropy_s_all, 'entropy_t':entropy_t_all, 'distance':distance_all,'distance_t':distance_t_all, 'distance_s':distance_s_all, 'false_negative_distance':false_negative_distance, 'false_negative_entropy':false_negative_entropy}
                        # np.save('experiment',dict)
                        # print(np.load('experiment.npy',allow_pickle=True))



                          
                                


            formatter = "{0:.8f}"
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                recall1 += u_recall1[j]
                recall5 += u_recall5[j]
                recall10 += u_recall10[j]
                average_precision += u_average_precision[j]

                if self.setting.report_user > 0 and (j + 1) % self.setting.report_user == 0:
                    print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1',
                          formatter.format(u_recall1[j] / u_iter_cnt[j]), 'MAP',
                          formatter.format(u_average_precision[j] / u_iter_cnt[j]), sep='\t')

            # print('recall@1:', formatter.format(recall1 / iter_cnt))
            # print('recall@5:', formatter.format(recall5 / iter_cnt))
            # print('recall@10:', formatter.format(recall10 / iter_cnt))
            # print('MAP', formatter.format(average_precision / iter_cnt))
            # print('predictions:', iter_cnt)

            log_string(self._log, 'recall@1: ' + formatter.format(recall1 / iter_cnt))
            log_string(self._log, 'recall@5: ' + formatter.format(recall5 / iter_cnt))
            log_string(self._log, 'recall@10: ' + formatter.format(recall10 / iter_cnt))
            log_string(self._log, 'MAP: ' + formatter.format(average_precision / iter_cnt))
            print('predictions:', iter_cnt)
            dict = {'confidence_s':confidence_s_all, 'confidence_t':confidence_t_all, 'entropy_s':entropy_s_all, 'entropy_t':entropy_t_all, 'distance':distance_all,'distance_t':distance_t_all, 'distance_s':distance_s_all, 'false_negative_distance':false_negative_distance, 'false_negative_entropy':false_negative_entropy}
            np.save('experiment_pad1',dict)