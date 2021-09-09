"""Get norms of RoBERTa model over trajectory."""

import torch
from transformers import AutoModel, RobertaModel
from collections import defaultdict
import os
from tqdm import tqdm
from math import sqrt

DATA = os.environ["DATA"]


norm_labels = {
    "min": "min",
    "l2": "Parameter norm",
    "proj": R"$\mathrm{cos}(\delta_t, \theta_t)$",  # Alignment
    "delta": R"$\Vert \delta_t \Vert^2$",  # Step size
    "dir": R"$\mathrm{cos}(\theta_t, \theta_{t+1})$",
}


def get_model(fname: str) -> RobertaModel:
    os.system(f"tar -xvf {DATA}/checkpoints/{fname}.tar")
    model = AutoModel.from_pretrained(fname)
    os.system(f"rm -rf {fname}")
    return model


def get_params(model):
    return [p for n, p in model.named_parameters() if n.startswith("encoder.")]


def get_norms(params):
    flat_params = torch.cat([p.flatten() for p in params])
    return {
        "l2": flat_params.norm(p=2).item(),
        "min": min((p / sqrt(p.numel())).norm(p=2).item() for p in params),
    }


def cos_sim(x, y):
    return x @ y / (x.norm(p=2) + y.norm(p=2) + 1e-9)


checkpoints = sorted(
    [
        fname.replace(".tar", "")
        for fname in os.listdir(f"{DATA}/checkpoints")
        if fname.startswith("checkpoint-")
    ],
    key=lambda fname: int(fname.replace("checkpoint-", "")),
)
ckpts = [int(fname.replace("checkpoint-", "")) for fname in checkpoints]

norms = defaultdict(list)
last_params = None
for fname in tqdm(checkpoints):
    print("=" * 10, fname, "=" * 10)
    model = get_model(fname)
    params = get_params(model)
    for name, value in get_norms(params).items():
        norms[name].append(value)
    
    # This stuff computes alignment metrics.
    params = torch.cat([p.flatten() for p in params])
    if last_params is not None:
        step = params - last_params
        norms["delta"].append(step.norm(p=2).item())
        norms["proj"].append(cos_sim(step, last_params).item())
        norms["dir"].append(cos_sim(params, last_params).item())
    last_params = params

import matplotlib.pyplot as plt

if not os.path.isdir("figs"):
    os.makedirs("figs")
for name, values in norms.items():
    plt.figure()
    if len(values) < len(ckpts):
        plt.plot(ckpts[:-1], values, marker=".")
    else:
        plt.plot(ckpts, values, marker=".")
    plt.xlabel("Checkpoint $t$")
    plt.ylabel(norm_labels[name])
    plt.tight_layout()
    plt.savefig(f"figs/{name}.pdf")
    print(f"Saved figs/{name}.pdf")
