from __future__ import annotations

import os
import sys
sys.path.append(os.getcwd() + '/data_preparation')
import logging
logger = logging.getLogger("qet-predictor")

from pathlib import Path
from typing import TYPE_CHECKING, Any

import copy
import pickle

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from utils import calc_supermarq_features
from qiskit import QuantumCircuit

if TYPE_CHECKING:
    from numpy._typing import NDArray


def get_path_training_data():
    """Returns the path to the training data folder."""
    return Path(os.getcwd()) / "data"


def get_openqasm_gates():
    """Returns a list of all quantum gates within the openQASM 2.0 standard header."""
    # according to https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/qelib1.inc
    return [
        "u3",
        "u2",
        "u1",
        "cx",
        "id",
        "u0",
        "u",
        "p",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "sx",
        "sxdg",
        "cz",
        "cy",
        "swap",
        "ch",
        "ccx",
        "cswap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cp",
        "cu3",
        "csx",
        "cu",
        "rxx",
        "rzz",
        "rccx",
        "rc3x",
        "c3x",
        "c3sqrtx",
        "c4x",
        "xx_plus_yy",
        "ecr",
    ]


def dict_to_featurevector(gate_dict: dict[str, int]):
    """Calculates and returns the feature vector of a given quantum circuit gate dictionary."""
    res_dct = dict.fromkeys(get_openqasm_gates(), 0)
    for key, val in dict(gate_dict).items():
        if key in res_dct:
            res_dct[key] = val
    return res_dct


PATH_LENGTH = 260


def create_feature_dict(qc: str | QuantumCircuit):
    """Creates and returns a feature dictionary for a given quantum circuit."""
    if not isinstance(qc, QuantumCircuit):
        if len(qc) < PATH_LENGTH and Path(qc).exists():
            qc = QuantumCircuit.from_qasm_file(qc)
        elif "OPENQASM" in qc:
            qc = QuantumCircuit.from_qasm_str(qc)
        else:
            raise ValueError("Invalid input for 'qc' parameter.") from None

    ops_list = qc.count_ops()
    ops_list_dict = dict_to_featurevector(ops_list)

    feature_dict = {}
    for key in ops_list_dict:
        feature_dict[key] = float(ops_list_dict[key])

    feature_dict["num_qubits"] = float(qc.num_qubits)
    feature_dict["depth"] = float(qc.depth())

    supermarq_features = calc_supermarq_features(qc)
    feature_dict["program_communication"] = supermarq_features.program_communication
    feature_dict["critical_depth"] = supermarq_features.critical_depth
    feature_dict["entanglement_ratio"] = supermarq_features.entanglement_ratio
    feature_dict["parallelism"] = supermarq_features.parallelism
    feature_dict["liveness"] = supermarq_features.liveness
    return feature_dict


def refine_training_data(sample, non_zero_indices=None):
    training_data = copy.deepcopy(sample)
    if non_zero_indices is None:
        global_features_list = np.array([np.array(td.global_features) for td in training_data])
        non_zero_indices = [
            i for i in range(global_features_list.shape[1])
            if global_features_list[:, i].sum() > 0
        ]
        print("non_zero_indices:", non_zero_indices)
    for td in training_data:
        td.global_features = torch.tensor(np.array(td.global_features)[non_zero_indices])
    return training_data, non_zero_indices


def standardization_training_data(sample, stats=None):
    training_data = copy.deepcopy(sample)
    if stats is None:
        x = torch.cat([td.x for td in training_data])
        global_features = torch.cat([td.global_features for td in training_data])
        stats = {
            "means_x":  x.mean(0),
            "stds_x":   x.std(0),
            "means_gf": global_features.mean(0),
            "stds_gf":  global_features.std(0),
        }
    means_x  = stats["means_x"]
    stds_x   = stats["stds_x"]
    means_gf = stats["means_gf"]
    stds_gf  = stats["stds_gf"]
    for td in training_data:
        td.x              = (td.x - means_x) / (1e-8 + stds_x)
        td.global_features = (td.global_features - means_gf) / (1e-8 + stds_gf)
    return training_data, stats


def save_training_data(training_data: list):
    """Saves the given training data to the training data folder."""
    with open(get_path_training_data() / "training_data.npy", "wb") as f:
        pickle.dump(training_data, f)


def load_training_data():
    """Loads and returns the training data from the training data folder."""
    with open(get_path_training_data() / "training_data.npy", "rb") as f:
        return pickle.load(f)


def calc(y_true, y_pred):
    r_squared = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    nmse = mse / np.var(y_true)
    print("MSE:", mse)
    print("R-squared:", r_squared)
    print("NMSE:", nmse)
