import random
from tqdm import tqdm

import os
import random
from collections import defaultdict

import gensim
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pickle
from treelib import Node, Tree
from multiprocessing import Pool
from numba import jit
from functools import partial

class Node2Vec:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    SUBSEQUENCE = 'sub_sequence'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph: nx.Graph, dimensions: int = 10, walk_length: int = 3, num_walks: int = 1, p: float = 1,
                 q: float = 1, weight_key: str = 'weight', workers: int = 1, sampling_strategy: dict = None,
                 quiet: bool = False, temp_folder: str = None, seed: int = None):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param p: Return hyper parameter (default: 1)
        :param q: Inout parameter (default: 1)
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :param workers: Number of workers for parallel execution (default: 1)
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        :param seed: Seed for the random number generator.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        """

        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._precompute_probabilities()
        # self.walks = self._generate_walks()
    
    
    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:
            # Init tree structure
            # if source > 2:
            #     break
            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()
            
            if self.SUBSEQUENCE not in d_graph[source]:
                d_graph[source][self.SUBSEQUENCE] = dict()

            for current_node in self.graph.neighbors(source):
                # Assign the unnormalized sampling strategy weight, normalize during random walk
                #unnormalized_weights = list()
                #d_neighbors = list()
                # tree = Tree()
                # root = tree.create_node(source, source, data=1) #root node

                for destination in self.graph.neighbors(source):
                    if destination == current_node:
                        continue

                    # # Init probabilities dictff
                    # if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    #     d_graph[current_node][self.PROBABILITIES_KEY] = dict()
                    
                    # p = self.sampling_strategy[source].get(self.P_KEY,
                    #                                             self.p) if source in self.sampling_strategy else self.p
                    # q = self.sampling_strategy[source].get(self.Q_KEY,
                    #                                             self.q) if source in self.sampling_strategy else self.q
                    #try:
                    # if self.graph[source][destination].get(self.weight_key):
                    #     weight = self.graph[source][destination].get(self.weight_key, 1)
                    # else:
                    #     ## Example : AtlasView({0: {'type': 1, 'weight':0.1}})- when we have edge weight
                    #     edge = list(self.graph[source][destination])[-1]
                    #     weight = self.graph[source][destination][edge].get(self.weight_key, 1)
                    # # except:
                    # #     weight = 1
                    #ss_weight = weight

                    
                    
                    
                    # if destination == source:  # Backwards probability
                    #     ss_weight = weight * 1 / p
                    # elif destination in self.graph[source]:  # If the neighbor is connected to the source
                    #     ss_weight = weight
                    # else:
                    #     ss_weight = weight * 1 / q

                    # source->current_node
                    ss_weight = self.graph[source][current_node].get(self.weight_key)
                    threshold = 0.80
                    if ss_weight <= threshold:
                        #print(source, current_node, ss_weight)
                        continue
                    #unnormalized_weights.append(ss_weight)
                    if self.walk_length == 1:
                        if destination in self.graph.neighbors(current_node):
                            ss_weight1 = self.graph[current_node][destination].get(self.weight_key, 1)
                            if ss_weight1 <= threshold:
                                continue
                            if destination not in d_graph[source][self.SUBSEQUENCE]:
                                d_graph[source][self.SUBSEQUENCE][destination] = []
                            weights_sum = np.array([ss_weight, ss_weight1])
                            d_graph[source][self.SUBSEQUENCE][destination].append([weights_sum.mean(), current_node])
                            
                    if self.walk_length == 2:
                        if destination in self.graph.neighbors(current_node):
                            ss_weight1 = self.graph[current_node][destination].get(self.weight_key, 1)
                            if ss_weight1 <= threshold:
                                continue
                            if destination not in d_graph[source][self.SUBSEQUENCE]:
                                d_graph[source][self.SUBSEQUENCE][destination] = []
                            weights_sum = np.array([ss_weight, ss_weight1])
                            d_graph[source][self.SUBSEQUENCE][destination].append([weights_sum.mean(), current_node])
                                
                        for current_node1 in self.graph.neighbors(current_node):
                            if current_node1 == source:
                                continue
                            if destination in self.graph.neighbors(current_node1):
                                weight1 = self.graph[current_node1][destination].get(self.weight_key, 1)
                                weight2 = self.graph[current_node][current_node1].get(self.weight_key, 1)
                                if weight1 >= threshold and weight2 >= threshold:
                                    weights_sum = np.array([ss_weight, weight1, weight2])
                                    if destination not in d_graph[source][self.SUBSEQUENCE]:
                                        d_graph[source][self.SUBSEQUENCE][destination] = []
                                    d_graph[source][self.SUBSEQUENCE][destination].append([weights_sum.mean(), current_node, current_node1])
                    
                    if self.walk_length == 3:
                        if destination in self.graph.neighbors(current_node):
                            ss_weight1 = self.graph[current_node][destination].get(self.weight_key, 1)
                            if ss_weight1 <= threshold:
                                continue
                            if destination not in d_graph[source][self.SUBSEQUENCE]:
                                d_graph[source][self.SUBSEQUENCE][destination] = []
                            weights_sum = np.array([ss_weight, ss_weight1])
                            d_graph[source][self.SUBSEQUENCE][destination].append([weights_sum.mean(), current_node])
                            
                        
                        for current_node1 in self.graph.neighbors(current_node):
                            if current_node1 == source:
                                continue
                            if destination in self.graph.neighbors(current_node1):
                                weight1 = self.graph[current_node1][destination].get(self.weight_key, 1)
                                weight2 = self.graph[current_node][current_node1].get(self.weight_key, 1)
                                if weight1 >= threshold and weight2 >= threshold:
                                    weights_sum = np.array([ss_weight, weight1, weight2])
                                    if destination not in d_graph[source][self.SUBSEQUENCE]:
                                        d_graph[source][self.SUBSEQUENCE][destination] = []
                                    d_graph[source][self.SUBSEQUENCE][destination].append([weights_sum.mean(), current_node, current_node1])
                            
                            for current_node2 in self.graph.neighbors(current_node1):
                                if current_node2 == source or current_node2 == current_node:
                                    continue
                                if destination in self.graph.neighbors(current_node2):
                                    weight1 = self.graph[current_node2][destination].get(self.weight_key, 1)
                                    weight2 = self.graph[current_node1][current_node2].get(self.weight_key, 1)
                                    weight3 = self.graph[current_node][current_node1].get(self.weight_key, 1)
                                    if weight1 >= threshold and weight2 >= threshold and weight3 >= threshold:
                                        weights_sum = np.array([ss_weight, weight1, weight2, weight3])
                                        if destination not in d_graph[source][self.SUBSEQUENCE]:
                                            d_graph[source][self.SUBSEQUENCE][destination] = []
                                        d_graph[source][self.SUBSEQUENCE][destination].append([weights_sum.mean(), current_node, current_node1, current_node2])

                                        

                # if destination in d_graph[source][self.SUBSEQUENCE]:
                #     # Normalize
                #     unnormalized_weights = np.array(unnormalized_weights)
                #     for t in range(len(d_graph[source][self.SUBSEQUENCE][destination])):
                #         d_graph[source][self.SUBSEQUENCE][destination][t][0] = d_graph[source][self.SUBSEQUENCE][destination][t][0] / unnormalized_weights.sum()
                        #print(d_graph[source][self.SUBSEQUENCE][current_node][t][0], unnormalized_weights.sum())




                    # print(len(unnormalized_weights), d_graph[source][self.SUBSEQUENCE][current_node], source, current_node)
                # print(d_graph[source][self.SUBSEQUENCE][current_node])
                # print(source, current_node, destination)
                #     all_nodes = [node.identifier for node in tree.all_nodes_itr()] 
                #     if destination not in all_nodes:
                #         tree.create_node(destination, destination, parent=source, data=ss_weight)
                # #tree.show()
                
                # for i in range(self.walk_length-1):
                #     tree = self.dfs_func(current_node, tree, i)
                # #tree.show()

                # sub_sequence = []
                # sub_weight = []
                # path_sequence = tree.paths_to_leaves()
                # for sub_path in path_sequence:
                #     if sub_path[-1] == current_node:
                #         sub_sequence.append(sub_path[1:-1])

                # for sub_path in sub_sequence:
                #     if len(sub_sequence) == 1:
                #         sub_weight.append(tree[sub_path[0]].data)
                #     else:
                #         w = 1
                #         for single in sub_path:
                #             w = w * tree[single].data
                #         sub_weight.append(w)
                # #print(sub_sequence, sub_weight, current_node)

            
    

            # # Calculate first_travel weights for source
            # first_travel_weights = []

            # for destination in self.graph.neighbors(source):
            #     first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))

            # first_travel_weights = np.array(first_travel_weights)
            # d_graph[source][self.FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()

            # # Save neighbors
            # d_graph[source][self.NEIGHBORS_KEY] = list(self.graph.neighbors(source))
            #del tree

        graph_file = "./gowalla_80.pkl"
        # with open(graph_file, 'wb') as f:
        #     pickle.dump(d_graph, f, protocol=2)
        print("prompt_set:", graph_file)
        with open(graph_file, 'rb') as f:  
            d_graph = pickle.load(f)  
        self.d_graph = d_graph
        # print("successfully load the d_graph!")
    
    @jit
    def dfs_func(self, current_node, tree, num_walk):
        # Calculate unnormalized weights
        leaves = [node.identifier for node in tree.leaves()]
        for source in leaves:
            for destination in self.graph.neighbors(source):
                p = self.sampling_strategy[source].get(self.P_KEY,
                                                            self.p) if source in self.sampling_strategy else self.p
                q = self.sampling_strategy[source].get(self.Q_KEY,
                                                            self.q) if source in self.sampling_strategy else self.q
                #try:
                if self.graph[source][destination].get(self.weight_key):
                    weight = self.graph[source][destination].get(self.weight_key, 1)
                else: 
                    ## Example : AtlasView({0: {'type': 1, 'weight':0.1}})- when we have edge weight
                    edge = list(self.graph[source][destination])[-1]
                    weight = self.graph[source][destination][edge].get(self.weight_key, 1)
                # except:
                #     weight = 1 
                
                ss_weight = weight
                # if destination == source:  # Backwards probability
                #     ss_weight = weight * 1 / p
                # elif destination in self.graph[source]:  # If the neighbor is connected to the source
                #     ss_weight = weight
                # else:
                #     ss_weight = weight * 1 / q
                all_nodes = [node.identifier for node in tree.all_nodes_itr()]
                if destination in all_nodes:
                    continue
                if destination == current_node:
                    tree.create_node(destination, destination, parent=source, data=ss_weight)
                    continue
                if (self.walk_length - 2) == num_walk and destination != current_node:
                    continue
                tree.create_node(destination, destination, parent=source, data=ss_weight)

        return tree


    def _generate_walks(self, start_node, end_node, direction, walk_length) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(self.parallel_generate_walks)(self.d_graph,
                                             start_node,
                                             end_node,
                                             direction,
                                             walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))
        #self.direction: forward:True, reverse:False
        walks = flatten(walk_results)

        return walks

    def parallel_generate_walks(self, d_graph: dict, start_node: int, end_node: int, direction: bool, global_walk_length: int, num_walks: int, cpu_num: int,
                                sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                                neighbors_key: str = None, probabilities_key: str = None, first_travel_key: str = None,
                                quiet: bool = False) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        walks_all = list()

        # if not quiet:
        #     pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

        for n_walk in range(num_walks):
            walks = list()
            # # Update progress bar
            # if not quiet:
            #     pbar.update(1)

            # Shuffle the nodes
            #shuffled_nodes = list(d_graph.keys())
            #random.shuffle(shuffled_nodes)
            
            # Start a random walk from every node
            for index, source in enumerate(start_node):
                # Skip nodes with specific num_walks
                # if source in sampling_strategy and \
                #         num_walks_key in sampling_strategy[source] and \
                #         sampling_strategy[source][num_walks_key] <= n_walk:
                #     continue

                # Start walk
                walk = [source]
                walk_options = []
                walk_weight = []

                if end_node[index] in d_graph[walk[-1]][self.SUBSEQUENCE]:
                    for t in d_graph[walk[-1]][self.SUBSEQUENCE][end_node[index]]:
                        if len(t) == 2:
                            walk_options.append([t[1]])
                            walk_weight.append(t[0])
                        elif len(t) == 3:
                            walk_options.append([t[1], t[2]])
                            walk_weight.append(t[0])
                        elif len(t) == 4:
                            walk_options.append([t[1], t[2], t[3]])
                            walk_weight.append(t[0])
                    walk_to = random.choices(walk_options, weights=walk_weight)[0]
                    
                else:
                    walk_to = []
                walks.append(walk_to)
            walks_all.append(walks)

            #     # Perform walk  direction
            #     while len(walk) < walk_length:
            #         if direction:
            #             walk_options = d_graph[walk[-1]].get(neighbors_key, None)
            #         else:
            #             walk_options = d_graph[walk[0]].get(neighbors_key, None)

            #         # Skip dead end nodes
            #         if not walk_options:
            #             break

            #         if len(walk) == 1:  # For the first step
            #             if direction:
            #                 probabilities = d_graph[walk[-1]][first_travel_key]
            #             else:
            #                 probabilities = d_graph[walk[0]][first_travel_key]
            #             walk_to = random.choices(walk_options, weights=probabilities)[0]
            #         else:
            #             if direction:
            #                 probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
            #             else:
            #                 probabilities = d_graph[walk[0]][probabilities_key][walk[1]]
                        
            #             walk_to = random.choices(walk_options, weights=probabilities)[0]
            #         if direction:
            #             walk.append(walk_to)
            #         else:
            #             walk.insert(0, walk_to)
            #         if walk_to == end_node:
            #             break

            #     #walk = list(map(str, walk))  # Convert all to strings
            #     if (len(walk) <= walk_length and walk[-1] == end_node):
            #         #walk.extend([106994]*(walk_length - len(walk)))  
            #         walks.append(walk[1:-1])
            #     else:
            #         walks.append([])
            # walks_all.append(walks)

        # if not quiet:
        #     pbar.close()

        return walks_all


    # def parallel_generate_walks(self, d_graph: dict, start_node: int, end_node: int, direction: bool, global_walk_length: int, num_walks: int, cpu_num: int,
    #                             sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
    #                             neighbors_key: str = None, probabilities_key: str = None, first_travel_key: str = None,
    #                             quiet: bool = False) -> list:
    #     """
    #     Generates the random walks which will be used as the skip-gram input.
    #     :return: List of walks. Each walk is a list of nodes.
    #     """

    #     walks_all = list()

    #     # if not quiet:
    #     #     pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    #     for n_walk in range(num_walks):
    #         walks = list()
    #         # # Update progress bar
    #         # if not quiet:
    #         #     pbar.update(1)

    #         # Shuffle the nodes
    #         #shuffled_nodes = list(d_graph.keys())
    #         #random.shuffle(shuffled_nodes)
            
    #         # Start a random walk from every node
    #         for source in start_node:
    #             # Skip nodes with specific num_walks
    #             if source in sampling_strategy and \
    #                     num_walks_key in sampling_strategy[source] and \
    #                     sampling_strategy[source][num_walks_key] <= n_walk:
    #                 continue

    #             # Start walk
    #             walk = [source]

    #             # Calculate walk length
    #             if source in sampling_strategy:
    #                 walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
    #             else:
    #                 walk_length = global_walk_length

    #             # Perform walk  direction
    #             while len(walk) < walk_length:
    #                 if direction:
    #                     walk_options = d_graph[walk[-1]].get(neighbors_key, None)
    #                 else:
    #                     walk_options = d_graph[walk[0]].get(neighbors_key, None)

    #                 # Skip dead end nodes
    #                 if not walk_options:
    #                     break

    #                 if len(walk) == 1:  # For the first step
    #                     if direction:
    #                         probabilities = d_graph[walk[-1]][first_travel_key]
    #                     else:
    #                         probabilities = d_graph[walk[0]][first_travel_key]
    #                     walk_to = random.choices(walk_options, weights=probabilities)[0]
    #                 else:
    #                     if direction:
    #                         probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
    #                     else:
    #                         probabilities = d_graph[walk[0]][probabilities_key][walk[1]]
                        
    #                     walk_to = random.choices(walk_options, weights=probabilities)[0]
    #                 if direction:
    #                     walk.append(walk_to)
    #                 else:
    #                     walk.insert(0, walk_to)
    #                 if walk_to == end_node:
    #                     break

    #             #walk = list(map(str, walk))  # Convert all to strings
    #             if (len(walk) <= walk_length and walk[-1] == end_node):
    #                 #walk.extend([106994]*(walk_length - len(walk)))  
    #                 walks.append(walk[1:-1])
    #             else:
    #                 walks.append([])
    #         walks_all.append(walks)

    #     # if not quiet:
    #     #     pbar.close()

    #     return walks_all
    
    def fit(self, **skip_gram_params) -> gensim.models.Word2Vec:
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameters for gensim.models.Word2Vec - do not supply 'size' / 'vector_size' it is
            taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        # Figure out gensim version, naming of output dimensions changed from size to vector_size in v4.0.0
        gensim_version = pkg_resources.get_distribution("gensim").version
        size = 'size' if gensim_version < '4.0.0' else 'vector_size'
        if size not in skip_gram_params:
            skip_gram_params[size] = self.dimensions

        if 'sg' not in skip_gram_params:
            skip_gram_params['sg'] = 1

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)