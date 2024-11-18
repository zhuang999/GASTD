import torch
import argparse
import sys

from network_s import RnnFactory


class Setting:
    """ Defines all settings in a single place using a command line interface.
    """

    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])  # foursquare has different default args.

        parser = argparse.ArgumentParser()
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        else:
            self.parse_gowalla(parser)
        self.parse_arguments(parser)
        args = parser.parse_args()

        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim  # 10
        self.mlp_hidden_dim = args.mlp_hidden_dim
        self.weight_decay = args.weight_decay  # 0.0
        self.learning_rate = args.lr  # 0.01
        self.epochs = args.epochs  # 100
        self.tea_epochs = args.tea_epochs
        self.rnn_factory = RnnFactory(args.rnn)  # RNN:0, GRU:1, LSTM:2
        self.is_lstm = self.rnn_factory.is_lstm()  # True or False
        self.lambda_t = args.lambda_t  # 0.01
        self.lambda_s = args.lambda_s  # 100 or 1000

        self.all_head_size = args.all_head_size
        self.all_head_size = args.all_head_size
        self.num_hidden_layers = args.num_hidden_layers
        self.num_attention_heads = args.num_attention_heads
        self.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.sample = args.sample

        self.id = args.id
        self.ts = args.ts
        self.dir = args.dir


        # data management
        self.dataset_file = './data/{}'.format(args.dataset)
        self.friend_file = './data/{}'.format(args.friendship)
        self.max_users = 0  # 0 = use all available users
        self.sequence_length = 20  # 
        self.batch_size = args.batch_size
        self.min_checkins = 101

        # evaluation        
        self.validate_epoch = args.validate_epoch  #
        self.report_user = args.report_user  # -1

        # log
        self.log_file = args.log_file

        self.trans_loc_file = args.trans_loc_file  # 
        self.trans_loc_spatial_file = args.trans_loc_spatial_file  # 
        self.trans_user_file = args.trans_user_file
        self.trans_interact_file = args.trans_interact_file

        self.lambda_user = args.lambda_user
        self.lambda_loc = args.lambda_loc

        self.use_weight = args.use_weight
        self.use_graph_user = args.use_graph_user
        self.use_spatial_graph = args.use_spatial_graph

        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    def parse_arguments(self, parser):
        # training
        parser.add_argument('--gpu', default=0, type=int, help='the gpu to use')  # -1
        parser.add_argument('--hidden-dim', default=10, type=int, help='hidden dimensions to use')  # 10
        parser.add_argument('--mlp-hidden-dim', default=10, type=int, help='hidden dimensions to use')  # 10
        parser.add_argument('--weight_decay', default=0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')  # 0.01
        parser.add_argument('--epochs', default=50, type=int, help='amount of epochs')  # 100
        parser.add_argument('--tea_epochs', default=35, type=int, help='amount of teacher epochs')  # 100
        parser.add_argument('--rnn', default='rnn', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')
        parser.add_argument('--num_hidden_layers', default=2, type=int, help='number of transformer hidden layers')  
        parser.add_argument('--num_attention_heads', default=2, type=int, help='number of transformer heads')  
        parser.add_argument('--attention_probs_dropout_prob', default=0.5, type=int, help='the ratio of dropout')  
        parser.add_argument('--hidden_dropout_prob', default=0.5, type=int, help='the ratio of dropout')
        parser.add_argument('--all_head_size', default=10, type=int, help='hidden dimensions in transformer hidden layers')  

        parser.add_argument('--sample', type=int, default=1, help='Sampling times of dropnode')
        parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
        parser.add_argument('--lam', type=float, default=1., help='Lamda') #0.0001


        # data management
        parser.add_argument('--dataset', default='checkins-gowalla.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')
        parser.add_argument('--friendship', default='gowalla_friend.txt', type=str,
                            help='the friendship file under ../data/<edges.txt> to load')
        # evaluation        
        parser.add_argument('--validate-epoch', default=5, type=int,
                            help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int,
                            help='report every x user on evaluation (-1: ignore)')

        # log
        parser.add_argument('--log_file', default='./results/log_gowalla', type=str,
                            help='存储结果日志')
        parser.add_argument('--trans_loc_file', default='./KGE/gowalla_scheme2_transe_loc_temporal_100.pkl', type=str,
                            help='使用transe方法构造的时间POI转换图') #'./KGE/gowalla_scheme2_transe_loc_temporal_100.pkl'
        parser.add_argument('--trans_user_file', default='', type=str,
                            help='使用transe方法构造的user转换图')
        parser.add_argument('--trans_loc_spatial_file', default='', type=str,
                            help='使用transe方法构造的空间POI转换图')
        parser.add_argument('--trans_interact_file', default='./KGE/gowalla_scheme2_transe_user-loc_100.pkl', type=str,
        #'./KGE/foursquare_scheme2_transe_user-loc_100.pkl'
                            help='使用transe方法构造的用户-POI交互图')
        parser.add_argument('--use_weight', default=False, type=bool, help='应用于GCN的AXW中是否使用W')
        parser.add_argument('--use_graph_user', default=False, type=bool, help='是否使用user graph')
        parser.add_argument('--use_spatial_graph', default=False, type=bool, help='是否使用空间POI graph')

        parser.add_argument(
        '-i', '--id',
        metavar='I',
        default='',
        help='The commit id)')
        parser.add_argument(
        '-t', '--ts',
        metavar='T',
        default='',
        help='The time stamp)')
        parser.add_argument(
        '-d', '--dir',
        metavar='D',
        default='',
        help='The output directory)')

    def parse_gowalla(self, parser):
        # defaults for gowalla dataset
        parser.add_argument('--batch-size', default=2, type=int,  # 200
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')

    def parse_foursquare(self, parser):
        # defaults for foursquare dataset
        parser.add_argument('--batch-size', default=800, type=int,
                            help='amount of users to process in one pass (batching)')  # 1024
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')

    def __str__(self):
        return (
                   'parse with foursquare default settings' if self.guess_foursquare else 'parse with gowalla default settings') + '\n' \
               + 'use device: {}'.format(self.device)
