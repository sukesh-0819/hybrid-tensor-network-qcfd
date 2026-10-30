import argparse
import json
from pathlib import Path

from qcfd.evolve import BurgersVQA
from qcfd.metrics import circuit_resources

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=8)
parser.add_argument("--encodings", nargs="+", default=["dense", "tn"])
args = parser.parse_args()

report = {}
for encoding in args.encodings:
    solver = BurgersVQA(args.N, T=0.2, encoding=encoding)
    result = solver.run(verbose=False)
    th_j, norm_j = result["step_inputs"][-1]
    circuit = solver.full_step_circuit(th_j, norm_j, result["theta_final"], result["infos"][-1]["phi"])
    report[encoding] = dict(qubits=circuit.num_qubits, **circuit_resources(circuit))
    print(encoding, {k: v for k, v in report[encoding].items() if k != "ops"})

(DATA / f"circuit_resources_N{args.N}.json").write_text(json.dumps(report, indent=2))
