import torch
import torch.nn.functional as F

class Graph:
    def __init__(self, node_feature_size, edge_feature_size, device):
        self.nodes = torch.zeros([0, node_feature_size], device=device, dtype=torch.float32)
        self.edge_source = torch.LongTensor([0], device=device, dtype=torch.long)
        self.edge_dest = torch.LongTensor([0], device=device, dtype=torch.long)
        self.edge_features = torch.LongTensor([0, edge_feature_size], device=device, dtype=torch.float32)
        self.owner_masks = None
        self.last_inserted_node = None

class Aggregator(torch.nn.Module):
    def __init__(self, state_size):
        super().__init__()

        self.transform = torch.nn.Linear(state_size, state_size)
        self.gate = torch.nn.Linear(state_size, state_size)

    def forward(self, graph: Graph):
        gates = self.gate(graph.nodes)
        data = self.transform(graph.nodes)

        return torch.mm(graph.owner_masks, data * gates)

def sample_softmax(tensor, dim=-1):
    res = F.gumbel_softmax(tensor, dim=dim)
    _, res = res.max(dim=dim)
    return res

def mask_softmax_input(tensor, mask):
    return torch.where(mask, tensor, torch.full([1], float("-inf"), dtype=tensor.dtype, device=tensor.device))

def masked_softmax(tensor, mask):
    tensor = mask_softmax_input(tensor, mask)
    return sample_softmax(tensor)

def masked_cross_entropy_loss(tensor, mask, target, enabled):
    tensor = mask_softmax_input(tensor, mask)
    l = F.cross_entropy(tensor, target, reduction="none")
    return torch.where(enabled, l, torch.zeros([1], dtype=l.dtype, device=l.device)).mean()

def remap_pad(t, pad_char):
    return torch.where(t != pad_char, t + 1, torch.zeros(1, dtype=t.dtype, device=t.device))


class EdgeAdder(torch.nn.Module):
    def __init__(self, state_size, aggregated_size, n_edge_dtypes, pad_char):
        super().__init__()

        self.pad_char = pad_char

        self.edge_decision_aggregator = Aggregator(state_size)
        self.edge_init = torch.nn.Parameter(torch.Tensor(n_edge_dtypes, state_size))
        self.edge_init_aggregator = Aggregator(state_size)

        self.f_addedge = torch.nn.Sequential(
            torch.nn.Linear(aggregated_size, n_edge_dtypes + 1),
            torch.nn.Softmax(-1)
        )

        self.fs_layer1_target = torch.nn.Linear(state_size, 1)
        self.fs_layer1_new = torch.nn.Linear(state_size, 1, bias=False)

    def forward(self, graph: Graph, reference, new_nodes: torch.Tensor):
        # Decide whether to add an edge.
        loss = 0
        new_edge_types = self.f_addedge(self.edge_decision_aggregator(graph))

        if reference is not None:
            selected_type = remap_pad(reference[1], self.pad_char)
            loss += F.cross_entropy(new_edge_types, selected_type, reduction="mean")
        else:
            selected_type = sample_softmax(new_edge_types)

        # Decide where to add
        # The transform is fs(new_node, all_other_nodes). First layer of this can be decomposed to
        # fs_layer1_target(all_other_nodes) + fs_layer1_new(new_node).

        logits = self.fs_layer1_target(graph.nodes).unsqueeze(0) + self.fs_layer1_new(new_nodes)

        if reference is not None:
            selected_other = reference[0]
            loss += masked_cross_entropy_loss(logits, graph.owner_masks, selected_other, selected_type!=0)
        else:
            selected_other = mask_softmax_input(logits, graph.owner_masks)

        return selected_type, selected_other, loss


class Propagator(torch.nn.Module):
    def __init__(self, state_size):
        super().__init__()

        self.message_size = state_size * 2

        self.node_update_fn = torch.nn.GRUCell(self.message_size, state_size)

        # The first layer of message function (fe) can be decomposed to 3 parts, which makes it easier to
        # claculate
        self.message_destnode = torch.nn.Linear(state_size, self.message_size)
        self.message_srcnode = torch.nn.Linear(state_size, self.message_size, bias=False)
        self.message_features = torch.nn.Linear(state_size, self.message_size, bias=False)

    def forward(self, graph: Graph):
        src_transformed  = self.message_srcnode(graph.nodes)
        dest_transformed  = self.message_srcnode(graph.nodes)

        edge_src = src_transformed.index_select(dim=0, index=graph.edge_source)
        edge_dest = dest_transformed.index_select(dim=0, index=graph.edge_dest)
        edge_features = self.message_features(graph.edge_features)

        messages = edge_src + edge_dest + edge_features
        # In case of multiple layers, subsequent layers should come here

        # Sum the messages for each node
        inputs = torch.zeros(graph.nodes.shape[0], self.message_size, device=graph.nodes.device,
                             dtype=graph.nodes.dtype)
        inputs.index_add_(0, graph.edge_dest, messages)

        # Transform node state
        graph.nodes = self.node_update_fn(inputs, graph.nodes)
        return graph


class MultilayerPropagator(torch.nn.Module):
    def __init__(self, state_size, n_steps):
        super().__init__()
        self.propagators = torch.nn.ModuleList([Propagator(state_size) for i in range(n_steps)])

    def forward(self, graph: Graph):
        for p in self.propagators:
            graph = p(graph)
        return graph


class GraphGen(torch.nn.Module):
    def __init__(self, n_node_types, edge_types, state_size, edge_feature_size, pad_char=255):
        super().__init__()

        self.state_size = state_size
        self.aggregated_size = self.state_size * 2
        self.edge_feature_size = edge_feature_size

        self.f_addnode = torch.nn.Sequential(
            torch.nn.Linear(self.aggregated_size, n_node_types + 1),
            torch.nn.Softmax(-1)
        )

        self.edge_adder = EdgeAdder(state_size, self.aggregated_size, edge_types, pad_char)



    def forward(self, data, ownership_mask):
        n_batch = data[0].shape[0]
        device = data[0].device

        loss = 0

        graph = Graph(self.state_size, self.edge_feature_size, device)


        aggregated_data = torch.zeros(n_batch, self.aggregated_size, device=device, dtype=torch.float32)
        # print(aggregated_data)

        for i in range(0, len(data), 2):
            # Decide whether to add node or not.
            node_prob = self.f_addnode(aggregated_data)
            loss += F.cross_entropy(node_prob, data[i], reduction="mean")

            edge_schema = data[i+1]
            for ei, e in enumerate(edge_schema):



