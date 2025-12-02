import torch.nn as nn
import torch

import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

class MatrixGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MatrixGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    def make_adjacency_matrix(self, edge_index, num_nodes):
        """
        Creates adjacency matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. dims: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: adjacency matrix with shape [num_nodes, num_nodes]

        Hint: A[i,j] -> there is an edge from node j to node i
        """
        # keep dtype consistent with node features (float32)
        adj = torch.zeros(
            (num_nodes, num_nodes),
            dtype=torch.float32,
            device=edge_index.device,
        )
        src_nodes, dst_nodes = edge_index
        # A[i, j] = 1 if j -> i
        adj[dst_nodes, src_nodes] = 1.0
        return adj

    def make_inverted_degree_matrix(self, edge_index, num_nodes):
        """
        Creates inverted degree matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: inverted degree matrix with shape [num_nodes, num_nodes]. Set degree of nodes without an edge to 1.
        """
        # in-degree: count incoming edges per node (destination index)
        dst_nodes = edge_index[1]
        deg = torch.bincount(dst_nodes, minlength=num_nodes).to(torch.float32)

        # nodes with zero in-degree should behave as if degree 1
        deg = deg.masked_fill(deg == 0, 1.0)

        inv_deg = 1.0 / deg
        inv_deg_mat = torch.diag(inv_deg).to(edge_index.device)
        return inv_deg_mat

    def forward(self, x, edge_index):
        """
        Forward propagation for GCNs using efficient matrix multiplication.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: activations for the GCN
        """
        num_nodes = x.size(0)
        A = self.make_adjacency_matrix(edge_index, num_nodes)
        D_inv = self.make_inverted_degree_matrix(edge_index, num_nodes)

        # D^{-1} A X W^T + X B^T   (same update as in Q3.3.a)
        out = D_inv @ A @ x @ self.W.t() + x @ self.B.t()
        return out

class MessageGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MessageGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    @staticmethod
    def message(x, edge_index):
        """
        message step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: message vector with shape [num_nodes, num_in_features]. Messages correspond to the old node values.

        Hint: check out torch.Tensor.index_add function
        """
        src, dst = edge_index
        num_nodes = x.size(0)

        # messages are the current features of source nodes
        messages = x[src]  # [num_edges, num_features]

        # aggregate (sum) by destination node
        aggregated_messages = torch.zeros_like(x)
        aggregated_messages.index_add_(0, dst, messages)

        # compute in-degree per node for mean aggregation
        sum_weight = torch.bincount(dst, minlength=num_nodes).to(x.dtype)
        # avoid division by zero: nodes with no incoming edges keep their own value
        sum_weight = sum_weight.masked_fill(sum_weight == 0, 1.0)

        aggregated_messages = aggregated_messages / sum_weight.unsqueeze(-1)
        return aggregated_messages

    def update(self, x, messages):
        """
        update step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param messages: messages vector with shape [num_nodes, num_in_features]
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        """
        # linear transform of aggregated neighbor messages + skip connection through B
        x = messages @ self.W.t() + x @ self.B.t()
        return x

    def forward(self, x, edge_index):
        message = self.message(x, edge_index)
        x = self.update(x, message)
        return x

class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.a = nn.Parameter(torch.Tensor(out_features * 2))

        nn.init.xavier_uniform_(self.W)
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x, edge_index, debug=False):
        """
        Forward propagation for GATs.
        Follow the implementation of Graph attention networks (Veličković et al. 2018).

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param debug: used for tests
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        :return: debug data for tests:
                 messages -> messages vector with shape [num_nodes, num_out_features], i.e. Wh from Veličković et al.
                 edge_weights_numerator -> unnormalized edge weightsm i.e. exp(e_ij) from Veličković et al.
                 softmax_denominator -> per destination softmax normalizer
        """
        # add self-loops as in the original GAT paper
        edge_index, _ = add_self_loops(edge_index)
        src, dst = edge_index

        # linear projection of node features
        activations = x @ self.W.t()                     # [N, F_out]
        messages = activations                           # node-level messages (Wh)

        # gather edge-wise source/destination representations
        h_src = messages[src]                            # [E, F_out]
        h_dst = messages[dst]                            # [E, F_out]

        # concatenate along feature dimension: [h_i || h_j]
        attention_inputs = torch.cat([h_src, h_dst], dim=-1)  # [E, 2*F_out]

        # unnormalized attention coefficients e_ij
        e_ij = F.leaky_relu(attention_inputs @ self.a)   # [E]

        # softmax numerator: exp(e_ij)
        edge_weights_numerator = torch.exp(e_ij)         # [E]

        # weight source features by attention numerators
        weighted_messages = edge_weights_numerator.view(-1, 1) * h_src  # [E, F_out]

        # per-destination normalization term (softmax denominator)
        num_nodes = x.size(0)
        softmax_denominator = x.new_zeros(num_nodes)     # [N]
        softmax_denominator.index_add_(0, dst, edge_weights_numerator)
        softmax_denominator = softmax_denominator.clamp_min(1e-12)

        # aggregate weighted messages at destination nodes
        aggregated_messages = torch.zeros_like(activations)
        aggregated_messages.index_add_(0, dst, weighted_messages)
        aggregated_messages = aggregated_messages / softmax_denominator.unsqueeze(-1)
        if not debug:
            return aggregated_messages
        
        return aggregated_messages, {
            'edge_weights': edge_weights_numerator,
            'softmax_weights': softmax_denominator,
            'messages': h_src,
        }
