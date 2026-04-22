import string

import networkx as nx
import rustworkx as rx
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
from qiskit_aer import AerSimulator
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler.passes import RemoveFinalMeasurements
from torch_geometric.utils.convert import from_networkx
from helper import get_openqasm_gates
import json
import sys
if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

import logging
logger = logging.getLogger("qet-predictor")


GATE_DICT = {item: index for index, item in enumerate(get_openqasm_gates())}
NUM_ERROR_DATA = 4
NUM_NODE_TYPE = 2 + len(GATE_DICT)


def get_global_features(circ):
    data = torch.zeros((1, 6))
    data[0][0] = circ.depth()
    data[0][1] = circ.width()
    for key in GATE_DICT:
        if key in circ.count_ops():
            data[0][2 + GATE_DICT[key]] = circ.count_ops()[key]

    return data


def to_networkx(dag):
    """Returns a copy of the DAGCircuit in networkx format."""
    G = nx.MultiDiGraph()
    for node in dag.nodes():
        G.add_node(node)
    for src, dest, wire in dag.edges():
        G.add_edge(src, dest, wire=wire)
    return G


def networkx_torch_convert(dag, global_features, length):
    myedge = []
    for item in dag.edges:
        myedge.append((item[0], item[1]))
    G = nx.DiGraph()
    G.add_nodes_from(dag._node)
    G.add_edges_from(myedge)
    x = torch.zeros((len(G.nodes()), length))
    for idx, node in enumerate(G.nodes()):
        try:
            x[idx] = dag.nodes[node]["x"]
        except KeyError:
            pass  # node type not in GATE_DICT; feature vector stays zero
    G = from_networkx(G)
    G.x = x
    G.global_features = global_features
    return G


_FAKE_BACKEND_CACHE = {}

FAKE_BACKEND_MAP = {
    "fake_nairobi":    "FakeNairobiV2",
    "fake_toronto":    "FakeTorontoV2",
    "fake_washington": "FakeWashingtonV2",
    "fake_sherbrooke": "FakeSherbrooke",
}

def get_noise_dict(device_name, n_qubits=20):
    """Return T1/T2 dict for the given device.
    - 'aer' / 'aer_gpu' or unknown → zeros (ideal simulation)
    - 'fake_nairobi', 'fake_toronto', etc. → real values from backend properties
    """
    noise_dict = {"qubit": {}, "gate": {}}

    backend_cls_name = FAKE_BACKEND_MAP.get(str(device_name).lower())
    if backend_cls_name:
        if backend_cls_name not in _FAKE_BACKEND_CACHE:
            from qiskit_ibm_runtime.fake_provider import FakeNairobiV2, FakeTorontoV2, FakeWashingtonV2, FakeSherbrooke
            _map = {"FakeNairobiV2": FakeNairobiV2, "FakeTorontoV2": FakeTorontoV2,
                    "FakeWashingtonV2": FakeWashingtonV2, "FakeSherbrooke": FakeSherbrooke}
            _FAKE_BACKEND_CACHE[backend_cls_name] = _map[backend_cls_name]()
        backend = _FAKE_BACKEND_CACHE[backend_cls_name]
        props = backend.properties()
        n_qubits = backend.num_qubits
        for i in range(n_qubits):
            t1 = props.qubit_property(i, "T1")
            t2 = props.qubit_property(i, "T2")
            noise_dict["qubit"][i] = {
                "T1": float(t1[0]) if t1 else 0.0,
                "T2": float(t2[0]) if t2 else 0.0,
            }
    else:
        # ideal Aer — zeros
        for i in range(n_qubits):
            noise_dict["qubit"][i] = {"T1": 0.0, "T2": 0.0}

    return noise_dict


def data_generator(node, noise_dict):
    try:
        if isinstance(node, DAGInNode):
            qubit_idx = int(node.wire._index)
            return "in", [qubit_idx], [noise_dict["qubit"][qubit_idx]]

        elif isinstance(node, DAGOutNode):
            qubit_idx = int(node.wire._index)
            return "out", [qubit_idx], [noise_dict["qubit"][qubit_idx]]
        elif isinstance(node, DAGOpNode):
            name = node.name
            qargs = node.qargs
            qubit_list = []
            for qubit in qargs:
                qubit_list.append(qubit._index)
            mylist = [noise_dict["qubit"][qubit_idx] for qubit_idx in qubit_list]
            return name, qubit_list, mylist
        else:
            raise NotImplementedError("Unknown node type")
    except Exception as e:
        logger.warning(e)


def circ_to_dag_with_data(circ, device_name, global_features, n_qubit=10):
    # data format: [node_type(onehot)]+[qubit_idx(one or two-hot)]+[T1,T2,T1,T2]+[gate_idx]
    circ = circ.copy()
    circ = RemoveBarriers()(circ)
    circ = RemoveFinalMeasurements()(circ)

    dag = circuit_to_dag(circ)
    dag = to_networkx(dag)
    dag_list = list(dag.nodes())

    noise_dict = get_noise_dict(device_name)
    # print(noise_dict)

    used_qubit_idx_list = {}
    used_qubit_idx = 0
    for node in dag_list:
        if isinstance(node, DAGOpNode) and node.name == 'measure':
            continue
        result = data_generator(node, noise_dict)
        if result is None:
            continue
        node_type, qubit_idxs, noise_info = result
        if node_type == "in":
            succnodes = dag.succ[node]
            for succnode in succnodes:
                if isinstance(succnode, DAGOpNode) and succnode.name == 'measure':
                    dag.remove_node(node)
                    dag.remove_node(succnode)
                    continue
                result = data_generator(succnode, noise_dict)
                if result is None:
                    continue
                succnode_type, _, _ = result
                if succnode_type == "out":
                    dag.remove_node(node)
                    dag.remove_node(succnode)
    dag_list = list(dag.nodes())
    for node_idx, node in enumerate(dag_list):
        try:
            node_type, qubit_idxs, noise_info = data_generator(node, noise_dict)
        except Exception as e:
            print(device_name)
        for qubit_idx in qubit_idxs:
            if not qubit_idx in used_qubit_idx_list:
                used_qubit_idx_list[qubit_idx] = used_qubit_idx
                used_qubit_idx += 1
        data = torch.zeros(NUM_NODE_TYPE + n_qubit + NUM_ERROR_DATA + 1)
        if node_type == "in":
            data[0] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
            data[NUM_NODE_TYPE + n_qubit] = noise_info[0]["T1"]
            data[NUM_NODE_TYPE + n_qubit + 1] = noise_info[0]["T2"]
        elif node_type == "out":
            data[1] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
            data[NUM_NODE_TYPE + n_qubit] = noise_info[0]["T1"]
            data[NUM_NODE_TYPE + n_qubit + 1] = noise_info[0]["T2"]
        else:
            if node_type not in GATE_DICT:
                continue
            data[2 + GATE_DICT[node_type]] = 1
            if len(qubit_idxs) == 2:
                for i in range(len(qubit_idxs)):
                    data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[i]]] = 1
                    data[NUM_NODE_TYPE + n_qubit + 2 * i] = noise_info[i]["T1"]
                    data[NUM_NODE_TYPE + n_qubit + 2 * i + 1] = noise_info[i]["T2"]
        data[-1] = node_idx
        if node in dag.nodes():
            dag.nodes[node]["x"] = data
    mapping = dict(zip(dag, string.ascii_lowercase))
    dag = nx.relabel_nodes(dag, mapping)
    return networkx_torch_convert(dag, global_features, length=NUM_NODE_TYPE + n_qubit + NUM_ERROR_DATA + 1)