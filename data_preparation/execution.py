from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import os
import sys
import multiprocessing
sys.path.append(os.getcwd() + '/data_preparation')
import helper
import circ_dag_converter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shutil
from joblib import Parallel, delayed, load
import utils
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit_aer import AerSimulator
from qiskit_aer import AerSimulator


def _subprocess_execute(qasm_str, backend_name, shots, result_queue):
    """Module-level function that runs in a child process.
    Recreates the backend fresh so it can be hard-killed on timeout.
    """
    try:
        import sys, os
        sys.path.insert(0, os.getcwd() + '/data_preparation')
        sys.path.insert(0, os.getcwd())
        import helper as _h
        from qiskit import QuantumCircuit, transpile, qasm2
        from qiskit_aer import AerSimulator

        from qiskit_aer import AerSimulator
        provider = AerSimulator()
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        tqc = transpile(qc, provider, optimization_level=0)
        job = provider.run(tqc, shots=shots)
        result = job.result()
        if not result.success:
            result_queue.put(False)
            return
        time_taken = result.results[0].time_taken
        if time_taken == 0.0:
            result_queue.put(False)
            return
        result_queue.put(time_taken)
    except Exception as e:
        result_queue.put(False)


if TYPE_CHECKING:
    from numpy._typing import NDArray

plt.rcParams["font.family"] = "Times New Roman"

logger = logging.getLogger("qet-predictor")


class Execution:
    def __init__(self, logger_level: int = logging.INFO) -> None:
        logger.setLevel(logger_level)

    def calculate_execution_time(self, backend_name, shots=1024, timeout=300):
        filefolder = helper.get_path_training_data() / "quantum_circuits"
        max_qubits = 127  # AerSimulator supports up to 127 qubits
        result_dir = helper.get_path_training_data() / backend_name
        result_dir.mkdir(parents=True, exist_ok=True)
        (helper.get_path_training_data() / "quantum_circuits_new").mkdir(parents=True, exist_ok=True)

        for file in sorted(filefolder.iterdir()):
            if file.suffix != ".qasm":
                continue
            print(file, flush=True)
            try:
                qc = QuantumCircuit.from_qasm_file(str(file))
            except Exception as e:
                logger.warning(e)
                continue
            if qc.num_qubits > max_qubits:
                continue

            # Use a subprocess so we can hard-kill the Aer C++ threads on timeout
            result_queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=_subprocess_execute,
                args=(qasm2.dumps(qc), backend_name, shots, result_queue)
            )
            p.start()
            p.join(timeout)
            if p.is_alive():
                p.terminate()
                p.join(5)
                if p.is_alive():
                    p.kill()
                print(f"  Skipped (timeout after {timeout}s)", flush=True)
                continue

            execute_result = result_queue.get() if not result_queue.empty() else False
            if execute_result is False or execute_result is None:
                print(f"  Skipped (error)", flush=True)
                continue

            time_taken = execute_result
            data = pd.DataFrame([{'quantum_circuit': qasm2.dumps(qc), 'time_taken': time_taken}])
            print('time_taken = ', time_taken, flush=True)
            data.to_csv(str(result_dir / (file.stem + ".csv")), index=False)
            new_file = helper.get_path_training_data() / "quantum_circuits_new" / file.name
            shutil.move(str(file), str(new_file))


    def calculate_execution_time_real_device(self, backend):
        filefolder = helper.get_path_training_data() / ("quantum_circuits_" + backend)
        provider = helper.provider_dict[backend.upper()]['provider']
        max_qubits = helper.provider_dict[backend.upper()]['max_qubits']
        result_dir = str(helper.get_path_training_data() / backend) + "/"
        for file in filefolder.iterdir():
            if file.suffix != ".qasm":
                continue
            print(file)
            try:
                qc = QuantumCircuit.from_qasm_file(str(file))
            except Exception as e:
                logger.warning(e)
                continue
            if qc.num_qubits > max_qubits:
                continue
            execute_result = self.execute_circuit(qc, provider, shots=1024)
            if execute_result is not False:
                time_taken = execute_result
                data = pd.DataFrame([{'quantum_circuit': qasm2.dumps(qc), 'time_taken': time_taken}])
                print('result.time_taken = ', time_taken)
                data.to_csv(result_dir + file.stem + ".csv", index=False)
                new_file = str(helper.get_path_training_data() / ("quantum_circuits_" + backend + "_new")) + "/" + file.name
                shutil.move(str(file), new_file)


    def execute_circuit(self, qc, backend, shots=1024):
        """Run circuit on AerSimulator backend and return result.time_taken, or False on failure."""
        try:
            tqc = transpile(qc, backend)
            job = backend.run(tqc, shots=shots)
            result = job.result()
            if not result.success:
                logger.warning("Circuit execution failed (result not successful)")
                return False
            time_taken = result.results[0].time_taken
            if time_taken == 0.0:
                logger.warning("time_taken is 0.0, skipping")
                return False
            return time_taken
        except Exception as e:
            logger.warning(f"Circuit execution failed: {e}")
            return False

    def calculate_average_execution_time(self, device, num=3):
        source_path = helper.get_path_training_data()
        folder_1 = source_path / f"{device}_1"
        final_result = pd.DataFrame(columns=["quantum_circuit", "time_taken", "device"])

        for csv_file in sorted(folder_1.iterdir()):
            if csv_file.suffix != ".csv":
                continue
            data = pd.read_csv(csv_file)
            time_taken = data['time_taken'].copy()
            quantum_circuit = data['quantum_circuit']

            for i in range(2, num + 1):
                trial_csv = source_path / f"{device}_{i}" / csv_file.name
                if trial_csv.exists():
                    trial_data = pd.read_csv(trial_csv)
                    time_taken = time_taken + trial_data['time_taken']

            time_taken = time_taken / num
            row = pd.DataFrame({
                'quantum_circuit': quantum_circuit,
                'time_taken': time_taken,
                'device': device
            })
            final_result = pd.concat([final_result, row], ignore_index=True)

        final_result.to_csv(str(source_path / f"{device}.csv"), index=False)
        print(f"Saved {len(final_result)} rows to {device}.csv")


    def calculate_average_execution_time_temp(self, device, num, list):
        source_path = str(helper.get_path_training_data()) + '/'
        time_taken = 0
        final_result = pd.DataFrame(columns=["quantum_circuit", "time_taken", "device"])
        for filename in list:
            for i in range(1, num + 1):
                data = pd.read_csv(source_path + device + '_' + str(i) + '/' + filename.split('.')[0] + '.csv')
                time_taken += data['time_taken']
            time_taken /= num
            final_result = pd.concat([final_result, pd.DataFrame({'quantum_circuit': filename, 'time_taken': time_taken, 'device': device})], ignore_index=True)
        final_result.to_csv(source_path + device + '.csv', index=False)


    def generate_training_sample_execution_time(self, data):
        from circ_dag_converter import FAKE_BACKEND_MAP
        training_data, scores_list = [], []
        total = len(data)

        # Cache transpiled backends to avoid reloading each circuit
        _backend_cache = {}

        def _get_fake_backend(device_name):
            key = str(device_name).lower()
            if key not in _backend_cache and key in FAKE_BACKEND_MAP:
                from qiskit_ibm_runtime.fake_provider import FakeNairobiV2, FakeTorontoV2, FakeWashingtonV2, FakeSherbrooke
                _map = {"FakeNairobiV2": FakeNairobiV2, "FakeTorontoV2": FakeTorontoV2,
                        "FakeWashingtonV2": FakeWashingtonV2, "FakeSherbrooke": FakeSherbrooke}
                cls_name = FAKE_BACKEND_MAP[key]
                _backend_cache[key] = _map[cls_name]()
            return _backend_cache.get(key)

        for idx, items in enumerate(data.itertuples()):
            circ = items[1]
            time_taken = items[2]
            device = items[3]
            qc = QuantumCircuit.from_qasm_str(circ)

            # Transpile to fake backend's native gate set if applicable
            fake_backend = _get_fake_backend(device)
            if fake_backend is not None:
                qc = transpile(qc, fake_backend, optimization_level=0)

            global_features = helper.create_feature_dict(qc)
            circ_graph_feature = circ_dag_converter.circ_to_dag_with_data(
                qc, device, list(global_features.values()), n_qubit=127
            )
            training_data.append(circ_graph_feature)
            scores_list.append(time_taken)
            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                print(f"[{idx+1}/{total}] graph conversion progress", flush=True)
        return (training_data, scores_list)
