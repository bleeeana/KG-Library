import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import degree



class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_node_types,
                 act=F.relu, bias=True, composition_op='mult', **kwargs):
        super(CompGCNConv, self).__init__(aggr='add', **kwargs)

        self._is_loop = None
        self._is_reversed = None
        self._rel_idx = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_node_types = num_node_types
        self.act = act
        self.composition_op = composition_op

        # Матрицы весов для разных направлений связей
        self.weight_in = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_out = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_loop = nn.Parameter(torch.Tensor(in_channels, out_channels))

        # Матрицы преобразования для отношений: T_r ∈ ℝ^{num_relations × in_channels × out_channels}
        self.rel_transform = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))

        # Эмбеддинги типов узлов: E_t ∈ ℝ^{num_node_types × in_channels}
        self.node_type_embeddings = nn.Embedding(num_node_types, in_channels)

        # Проекции для типов узлов: P_t ∈ ℝ^{num_node_types × out_channels × out_channels}
        self.node_type_projections = nn.Parameter(torch.Tensor(num_node_types, out_channels, out_channels))

        self.layer_norm = nn.LayerNorm(out_channels)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_in)
        nn.init.xavier_uniform_(self.weight_out)
        nn.init.xavier_uniform_(self.weight_loop)
        
        for i in range(self.num_relations):
            nn.init.xavier_uniform_(self.rel_transform[i])
        
        nn.init.xavier_uniform_(self.node_type_embeddings.weight)
        
        for i in range(self.num_node_types):
            nn.init.xavier_uniform_(self.node_type_projections[i])
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index_dict, edge_type_dict, node_type_dict):
        # Добавление информации о типах узлов: h̃_i = h_i + e_{t_i}
        node_types = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        for node_idx, node_type in node_type_dict.items():
            node_types[node_idx] = node_type
        x = x + self.node_type_embeddings(node_types)  # Обогащение признаков

        # Агрегация сообщений для каждого типа отношения
        in_msgs = torch.zeros(x.size(0), self.weight_in.size(1), device=x.device)
        out_msgs = torch.zeros(x.size(0), self.weight_out.size(1), device=x.device)
        loop_msgs = torch.zeros(x.size(0), self.weight_loop.size(1), device=x.device)

        # Счетчики количества соседей для нормализации
        in_degrees = torch.ones(x.size(0), device=x.device)
        out_degrees = torch.ones(x.size(0), device=x.device)
        loop_count = torch.ones(x.size(0), device=x.device)

        for rel_type, edge_index in edge_index_dict.items():
            if rel_type[0] != 'entity' or rel_type[2] != 'entity':
                continue

            rel_name = rel_type[1]
            if rel_name not in edge_type_dict:
                continue

            rel_idx = edge_type_dict[rel_name]
            self._rel_idx = rel_idx
            self._is_reversed = ':reversed' in rel_name
            self._is_loop = 'loop' in rel_name

            # Message Passing: m_{j→i} = (h̃_j · T_r) · W_{dir}
            result = self.propagate(edge_index, x=x)

            # Определение типа сообщения и учет степени узла
            if self._is_loop:
                loop_msgs += result
                row, col = edge_index
                loop_deg = degree(col, x.size(0), dtype=x.dtype)
                loop_count += loop_deg
            elif self._is_reversed:
                out_msgs += result
                row, col = edge_index
                out_deg = degree(col, x.size(0), dtype=x.dtype)
                out_degrees += out_deg
            else:
                in_msgs += result
                row, col = edge_index
                in_deg = degree(col, x.size(0), dtype=x.dtype)
                in_degrees += in_deg

        in_degrees = in_degrees.view(-1, 1)
        out_degrees = out_degrees.view(-1, 1)
        loop_count = loop_count.view(-1, 1)

        in_msgs = in_msgs / in_degrees
        out_msgs = out_msgs / out_degrees
        loop_msgs = loop_msgs / loop_count

        in_msgs = self.layer_norm(in_msgs)
        out_msgs = self.layer_norm(out_msgs)
        loop_msgs = self.layer_norm(loop_msgs)

        combined_msgs = in_msgs + out_msgs + loop_msgs

        # Проекция по типам узлов: h_i^{out} = h_i^{agg} · P_{t_i}
        projected = torch.zeros_like(combined_msgs)
        for node_idx, node_type in node_type_dict.items():
            projected[node_idx] = torch.matmul(combined_msgs[node_idx],
                                               self.node_type_projections[node_type])

        if self.bias is not None:
            projected += self.bias

        return {"entity": F.relu(projected)}

    def message(self, x_j: Tensor) -> Tensor:

        weight = self.weight_loop if getattr(self, '_is_loop', False) else \
            self.weight_out if getattr(self, '_is_reversed', False) else \
                self.weight_in

        # Применение преобразования: m_j = (x_j · T_r) · W_{dir}
        rel_transform = self.rel_transform[getattr(self, '_rel_idx', 0)]
        return torch.matmul(torch.matmul(x_j, rel_transform), weight)

    def update(self, aggr_out):
        return aggr_out