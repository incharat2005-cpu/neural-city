import time
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from flood import load_graph
from traffic import road_class, BASE_LOAD, demand

rng = np.random.default_rng(0)
torch.manual_seed(0)

# ---------- 1. Road graph ----------
g = load_graph()
nodes = list(g.nodes)
index = {n: i for i, n in enumerate(nodes)}
N = len(nodes)

und = nx.Graph(g.to_undirected())
A = sp.csr_matrix(nx.to_scipy_sparse_array(und, nodelist=nodes, weight=None, format="csr"),
                  dtype=np.float32)
A.setdiag(0)
A.eliminate_zeros()
deg = np.asarray(A.sum(axis=1)).ravel()
A_rw = sp.diags(1.0 / np.where(deg == 0, 1, deg)) @ A          # neighbour average
A_loop = A + sp.identity(N, dtype=np.float32)
d_inv_sqrt = 1.0 / np.sqrt(np.asarray(A_loop.sum(axis=1)).ravel())
A_hat = (sp.diags(d_inv_sqrt) @ A_loop @ sp.diags(d_inv_sqrt)).tocoo()   # GCN normalisation
A_t = torch.sparse_coo_tensor(
    torch.tensor(np.vstack([A_hat.row, A_hat.col]), dtype=torch.long),
    torch.tensor(A_hat.data, dtype=torch.float32), A_hat.shape).coalesce()

# ---------- 2. Simulated traffic (7 days, every 15 minutes) ----------
load_sum, load_cnt = np.zeros(N), np.zeros(N)
for u, v, d in g.edges(data=True):
    b = BASE_LOAD.get(road_class(d), 0.3)
    for n in (u, v):
        load_sum[index[n]] += b
        load_cnt[index[n]] += 1
base = load_sum / np.maximum(load_cnt, 1)       # busier road types -> higher base congestion

STEPS_PER_DAY, DAYS, H = 96, 7, 2               # H = forecast horizon: 2 steps = 30 minutes
T = STEPS_PER_DAY * DAYS
hours = (np.arange(T) * 0.25) % 24
C = np.zeros((T, N), dtype=np.float32)
C[0] = base * demand(hours[0])
for t in range(T - 1):
    nxt = 0.6 * C[t] + 0.3 * (A_rw @ C[t]) + 0.1 * base * demand(hours[t + 1])
    nxt = nxt + rng.normal(0, 0.02, N)
    hit = rng.random(N) < 0.0005                # random incidents that spread to neighbours
    nxt[hit] += 0.6
    C[t + 1] = np.clip(nxt, 0, 2)
print("Simulated traffic:", T, "time steps x", N, "intersections")

# ---------- 3. Training samples ----------
LAGS = 4
samples = list(range(LAGS - 1, T - H))
S = len(samples)
X = np.zeros((S, N, LAGS + 2), dtype=np.float32)   # last 4 readings + time of day (sin, cos)
Y = np.zeros((S, N), dtype=np.float32)
for s, t in enumerate(samples):
    X[s, :, :LAGS] = C[t - LAGS + 1:t + 1].T
    X[s, :, LAGS] = np.sin(2 * np.pi * hours[t] / 24)
    X[s, :, LAGS + 1] = np.cos(2 * np.pi * hours[t] / 24)
    Y[s] = C[t + H]
split = int(S * 0.7)                            # first 70% of time to train, last 30% to test
Xtr, Ytr, Xte, Yte = X[:split], Y[:split], X[split:], Y[split:]
print("Training samples:", len(Xtr), " Test samples (unseen time):", len(Xte))


def scores(pred):
    return float(np.sqrt(np.mean((pred - Yte) ** 2))), float(np.mean(np.abs(pred - Yte)))


# ---------- 4. Baselines ----------
persist = Xte[:, :, LAGS - 1]                   # "traffic stays as it is now"
r_persist = scores(persist)

ridge = Ridge(alpha=1.0).fit(Xtr.reshape(-1, LAGS + 2), Ytr.reshape(-1))
ridge_pred = ridge.predict(Xte.reshape(-1, LAGS + 2)).reshape(Yte.shape)
r_ridge = scores(ridge_pred)


# ---------- 5. Graph neural network (GCN) ----------
def propagate(A, Hx):                           # Hx: (batch, N, features); mixes each node with neighbours
    B, n, f = Hx.shape
    out = torch.sparse.mm(A, Hx.permute(1, 0, 2).reshape(n, B * f))
    return out.reshape(n, B, f).permute(1, 0, 2)


class GCN(nn.Module):
    def __init__(self, f_in, hidden=32):
        super().__init__()
        self.l1 = nn.Linear(f_in, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, A, x):
        h = torch.relu(self.l1(propagate(A, x)))
        h = torch.relu(self.l2(propagate(A, h)))
        return self.out(h).squeeze(-1)


model = GCN(LAGS + 2)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
start = time.time()
for epoch in range(30):
    perm = torch.randperm(len(Xtr_t))
    total = 0.0
    for i in range(0, len(perm), 32):
        idx = perm[i:i + 32]
        loss = F.mse_loss(model(A_t, Xtr_t[idx]), Ytr_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/30  training error (MSE): {total / len(Xtr_t):.5f}  "
              f"({time.time() - start:.0f}s)")

model.eval()
with torch.no_grad():
    gnn_pred = np.concatenate([model(A_t, torch.tensor(Xte[i:i + 32])).numpy()
                               for i in range(0, len(Xte), 32)])
r_gnn = scores(gnn_pred)
torch.save(model.state_dict(), "gnn_traffic.pt")

# ---------- 6. Results ----------
print()
print(f"{'Method':32s}{'RMSE':>8s}{'MAE':>8s}")
for name, r in [("Persistence (no learning)", r_persist),
                ("Ridge regression (no graph)", r_ridge),
                ("Graph neural network (GCN)", r_gnn)]:
    print(f"{name:32s}{r[0]:8.4f}{r[1]:8.4f}")
best_base = min(r_persist[0], r_ridge[0])
print(f"GNN error vs best baseline: {100 * (best_base - r_gnn[0]) / best_base:+.1f}% (positive = better)")

# ---------- 7. Picture for the slides ----------
busy = int(np.argmax(base))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(["Persistence", "Ridge\n(no graph)", "GNN\n(GCN)"],
            [r_persist[0], r_ridge[0], r_gnn[0]], color=["#888888", "#4c78a8", "#e63946"])
axes[0].set_ylabel("Forecast error, RMSE (lower is better)")
axes[0].set_title("30-minute traffic forecast error on unseen time")
axes[1].plot(Yte[:96, busy], label="Actual", color="black")
axes[1].plot(gnn_pred[:96, busy], label="GNN forecast", color="#e63946")
axes[1].plot(persist[:96, busy], label="Persistence", color="#888888", linestyle="--")
axes[1].set_title("One busy intersection, first test day")
axes[1].set_xlabel("15-minute steps")
axes[1].set_ylabel("Congestion level")
axes[1].legend()
plt.tight_layout()
plt.savefig("gnn_results.png", dpi=130)
print("Saved gnn_results.png and gnn_traffic.pt")