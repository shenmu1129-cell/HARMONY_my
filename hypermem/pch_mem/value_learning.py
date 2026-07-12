"""
Offline value learning for PCH-Mem retrieval policy.

Implements:
- Behavior Cloning (BC) warm-start from teacher trajectories
- Conservative Q-Learning (CQL) for offline value estimation
- Lightweight Q-network with MLP architecture
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import (
    ActionType,
    HypergraphState,
    MDPAction,
    MDPState,
    PCHConfig,
    PolicyHyperedge,
    RetrievalTrajectory,
    TrajectoryStep,
)

# Try importing torch, fall back to numpy-only implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── NumPy-only Q-Network (fallback) ─────────────────────────────

class NumpyQNetwork:
    """Lightweight Q-network using numpy (no PyTorch dependency)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int, num_layers: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # Xavier-like init
        scale = math.sqrt(2.0 / (input_dim + hidden_dim))
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        scale2 = math.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        scale_out = math.sqrt(2.0 / (hidden_dim + num_actions))
        self.W_out = np.random.randn(hidden_dim, num_actions).astype(np.float32) * scale_out
        self.b_out = np.zeros(num_actions, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x -> Q-values for all actions."""
        h = np.maximum(0, x @ self.W1 + self.b1)
        h = np.maximum(0, h @ self.W2 + self.b2)
        return h @ self.W_out + self.b_out

    def get_params(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W_out, self.b_out]

    def set_params(self, params: List[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2, self.W_out, self.b_out = params

    def copy(self) -> "NumpyQNetwork":
        new = NumpyQNetwork(self.input_dim, self.hidden_dim, self.num_actions)
        new.set_params([p.copy() for p in self.get_params()])
        return new


# ── Torch Q-Network ──────────────────────────────────────────────

if HAS_TORCH:
    class QNetwork(nn.Module):
        """PyTorch Q-network for value learning."""

        def __init__(self, input_dim: int, hidden_dim: int, num_actions: int, num_layers: int = 2):
            super().__init__()
            layers = []
            in_dim = input_dim
            for i in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, num_actions))
            self.net = nn.Sequential(*layers)
            self.num_actions = num_actions

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def get_q_values(self, state_features: np.ndarray) -> np.ndarray:
            """Get Q-values for a single state (numpy interface)."""
            self.eval()
            with torch.no_grad():
                x = torch.from_numpy(state_features).float().unsqueeze(0)
                q = self.forward(x).squeeze(0).numpy()
            return q
else:
    # Alias for type hints
    QNetwork = NumpyQNetwork  # type: ignore


class ValuePolicy:
    """Value-based retrieval policy.

    Wraps a Q-network and provides action selection,
    training, and target network management.
    """

    def __init__(
        self,
        config: PCHConfig,
        hypergraph: HypergraphState,
    ):
        self.config = config
        self.hg = hypergraph

        # Compute input dimension from a dummy state
        dummy_state = MDPState(
            query_embedding=np.zeros(config.embedding_dim, dtype=np.float32),
            evidence_embedding=np.zeros(config.embedding_dim, dtype=np.float32),
        )
        feature_vec = dummy_state.to_feature_vector(config.embedding_dim)
        self.input_dim = len(feature_vec)
        self.num_actions = max(1, len(hypergraph.structural_edges) + 2)  # +2 for STOP, FALLBACK

        if HAS_TORCH:
            self.q_network = QNetwork(
                self.input_dim, config.hidden_dim, self.num_actions, config.q_hidden_layers
            )
            self.target_network = copy.deepcopy(self.q_network)
            self.optimizer = torch.optim.Adam(
                self.q_network.parameters(), lr=config.learning_rate
            )
            self._torch = True
        else:
            self.q_network = NumpyQNetwork(
                self.input_dim, config.hidden_dim, self.num_actions, config.q_hidden_layers
            )
            self.target_network = self.q_network.copy()
            self._torch = False

        # Action indexing
        self.action_to_idx: Dict[Tuple[str, str], int] = {}
        self.idx_to_action: Dict[int, MDPAction] = {}
        self._build_action_index()

    def _build_action_index(self) -> None:
        """Build mapping from MDP actions to Q-network output indices."""
        self.action_to_idx.clear()
        self.idx_to_action.clear()
        idx = 0

        # Policy edges first
        for eid in self.hg.policy_edges:
            action = MDPAction(ActionType.SELECT_POLICY, eid)
            key = (action.action_type.value, action.edge_id)
            self.action_to_idx[key] = idx
            self.idx_to_action[idx] = action
            idx += 1

        # Structural edges
        for eid in self.hg.structural_edges:
            action = MDPAction(ActionType.SELECT_STRUCT, eid)
            key = (action.action_type.value, action.edge_id)
            self.action_to_idx[key] = idx
            self.idx_to_action[idx] = action
            idx += 1

        # Terminal actions
        for action in [MDPAction(ActionType.STOP), MDPAction(ActionType.FALLBACK)]:
            key = (action.action_type.value, action.edge_id)
            self.action_to_idx[key] = idx
            self.idx_to_action[idx] = action
            idx += 1

        # Resize network output if needed
        self.num_actions = idx

    def get_q_values(self, state: MDPState) -> np.ndarray:
        """Get Q-values for all actions from a state."""
        features = state.to_feature_vector(self.config.embedding_dim)

        if self._torch:
            self.q_network.eval()
            with torch.no_grad():
                x = torch.from_numpy(features).float().unsqueeze(0)
                q = self.q_network(x).squeeze(0).numpy()
        else:
            q = self.q_network.forward(features)

        # Pad/trim to num_actions
        if len(q) < self.num_actions:
            q = np.pad(q, (0, self.num_actions - len(q)), constant_values=-1e6)
        return q[:self.num_actions]

    def select_action(self, state: MDPState, available_actions: List[MDPAction]) -> MDPAction:
        """Select best action from available actions using Q-values."""
        q_values = self.get_q_values(state)
        best_action = None
        best_q = -float("inf")

        for action in available_actions:
            key = (action.action_type.value, action.edge_id)
            idx = self.action_to_idx.get(key)
            if idx is not None and idx < len(q_values):
                if q_values[idx] > best_q:
                    best_q = q_values[idx]
                    best_action = action

        if best_action is None:
            best_action = MDPAction(ActionType.STOP)
        return best_action

    def select_action_batch(
        self,
        state_features: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Select best actions for a batch of states with action masks."""
        if self._torch:
            self.q_network.eval()
            with torch.no_grad():
                x = torch.from_numpy(state_features).float()
                q = self.q_network(x).numpy()
        else:
            q = np.stack([self.q_network.forward(sf) for sf in state_features])

        # Apply mask: -inf for unavailable actions
        masked_q = np.where(action_mask, q, -1e10)
        return np.argmax(masked_q, axis=1)

    def sync_target_network(self) -> None:
        """Sync target network with current Q-network."""
        if self._torch:
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            self.target_network = self.q_network.copy()

    def update_action_index(self) -> None:
        """Rebuild action index after topology changes."""
        old_num = self.num_actions
        self._build_action_index()
        if self.num_actions != old_num and self._torch:
            # Need to rebuild network with new output size
            new_network = QNetwork(
                self.input_dim, self.config.hidden_dim,
                self.num_actions, self.config.q_hidden_layers
            )
            # Copy old weights where possible
            # (simplified: just create new optimizer)
            self.q_network = new_network
            self.target_network = copy.deepcopy(new_network)
            self.optimizer = torch.optim.Adam(
                self.q_network.parameters(), lr=self.config.learning_rate
            )


# ── Training Functions ───────────────────────────────────────────


def train_bc(
    policy: ValuePolicy,
    trajectories: List[RetrievalTrajectory],
    config: PCHConfig,
) -> List[float]:
    """Behavior Cloning: imitate high-return teacher actions.

    L_BC = -E_{(s,a)~D} log pi(a|s)
    """
    # Collect (state, action) pairs from high-return trajectories
    samples: List[Tuple[np.ndarray, int]] = []

    # Filter to top trajectories by return
    if trajectories:
        returns = [t.total_return for t in trajectories]
        threshold = np.percentile(returns, 50) if len(returns) > 1 else -float("inf")
    else:
        threshold = -float("inf")

    for traj in trajectories:
        if traj.total_return < threshold:
            continue
        for step in traj.steps:
            features = step.state.to_feature_vector(config.embedding_dim)
            key = (step.action.action_type.value, step.action.edge_id)
            action_idx = policy.action_to_idx.get(key)
            if action_idx is not None:
                samples.append((features, action_idx))

    if not samples:
        return [0.0]

    losses = []
    for epoch in range(config.bc_epochs):
        # Mini-batch SGD
        np.random.shuffle(samples)
        epoch_loss = 0.0
        for i in range(0, len(samples), config.batch_size):
            batch = samples[i:i + config.batch_size]
            batch_features = np.stack([s[0] for s in batch])
            batch_targets = np.array([s[1] for s in batch], dtype=np.int64)

            if policy._torch:
                x = torch.from_numpy(batch_features).float()
                targets = torch.from_numpy(batch_targets).long()

                policy.q_network.train()
                policy.optimizer.zero_grad()
                logits = policy.q_network(x)
                # Cross-entropy loss (BC)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                policy.optimizer.step()
                epoch_loss += loss.item()
            else:
                # Simple numpy SGD
                lr = config.learning_rate
                for j in range(len(batch)):
                    x = batch_features[j]
                    target = batch_targets[j]
                    q = policy.q_network.forward(x)
                    # Softmax + cross-entropy
                    q_exp = np.exp(q - np.max(q))
                    probs = q_exp / q_exp.sum()
                    probs[target] -= 1.0  # gradient of cross-entropy

                    # Backprop (simplified: only output layer)
                    h = np.maximum(0, x @ policy.q_network.W1 + policy.q_network.b1)
                    h = np.maximum(0, h @ policy.q_network.W2 + policy.q_network.b2)
                    policy.q_network.W_out -= lr * np.outer(h, probs)
                    policy.q_network.b_out -= lr * probs
                    epoch_loss -= math.log(max(1e-8, 1.0 - abs(float(probs[target]))))

        losses.append(epoch_loss / max(1, len(samples)))
        if epoch > 5 and len(losses) > 1 and abs(losses[-1] - losses[-2]) < 1e-5:
            break

    policy.sync_target_network()
    return losses


def train_cql(
    policy: ValuePolicy,
    trajectories: List[RetrievalTrajectory],
    config: PCHConfig,
) -> List[float]:
    """Conservative Q-Learning for offline value estimation.

    L = L_TD + alpha_cql * L_CQL + alpha_bc * L_BC

    The CQL term penalizes overestimation of OOD actions.
    """
    if not trajectories:
        return [0.0]

    # Build replay buffer from all trajectory steps
    buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
    for traj in trajectories:
        for step in traj.steps:
            s = step.state.to_feature_vector(config.embedding_dim)
            key = (step.action.action_type.value, step.action.edge_id)
            a = policy.action_to_idx.get(key)
            if a is None:
                continue
            r = step.reward
            s_next = step.next_state.to_feature_vector(config.embedding_dim)
            d = step.done
            buffer.append((s, a, r, s_next, d))

    if not buffer:
        return [0.0]

    losses = []
    for epoch in range(config.cql_epochs):
        np.random.shuffle(buffer)
        epoch_loss = 0.0
        epoch_td = 0.0
        epoch_cql = 0.0

        for i in range(0, len(buffer), config.batch_size):
            batch = buffer[i:i + config.batch_size]
            batch_size = len(batch)

            batch_s = np.stack([b[0] for b in batch])
            batch_a = np.array([b[1] for b in batch], dtype=np.int64)
            batch_r = np.array([b[2] for b in batch], dtype=np.float32)
            batch_sn = np.stack([b[3] for b in batch])
            batch_d = np.array([b[4] for b in batch], dtype=np.float32)

            # Build action mask: 1 for available, 0 for unavailable
            # In offline setting, all actions in the index are "available"
            action_mask = np.ones((batch_size, policy.num_actions), dtype=np.float32)

            if policy._torch:
                s_t = torch.from_numpy(batch_s).float()
                a_t = torch.from_numpy(batch_a).long()
                r_t = torch.from_numpy(batch_r).float()
                sn_t = torch.from_numpy(batch_sn).float()
                d_t = torch.from_numpy(batch_d).float()

                policy.q_network.train()
                policy.optimizer.zero_grad()

                # Current Q-values
                q_all = policy.q_network(s_t)
                q_current = q_all.gather(1, a_t.unsqueeze(1)).squeeze(1)

                # Target Q-values
                with torch.no_grad():
                    q_next = policy.target_network(sn_t)
                    q_next_max = q_next.max(dim=1)[0]
                    q_target = r_t + config.gamma * q_next_max * (1 - d_t)

                # TD loss
                td_loss = F.mse_loss(q_current, q_target)

                # CQL loss: penalize high Q-values for all actions,
                # encourage Q-values for actions in the dataset
                cql_loss = torch.logsumexp(q_all, dim=1).mean() - q_current.mean()

                # BC regularization
                bc_loss = F.cross_entropy(q_all, a_t)

                # Combined loss
                loss = td_loss + config.cql_alpha * cql_loss + config.bc_alpha * bc_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.q_network.parameters(), 1.0)
                policy.optimizer.step()

                epoch_loss += loss.item()
                epoch_td += td_loss.item()
                epoch_cql += cql_loss.item()
            else:
                # Simplified numpy training loop
                for j in range(batch_size):
                    s = batch_s[j]
                    a = batch_a[j]
                    r = batch_r[j]
                    sn = batch_sn[j]
                    d = batch_d[j]

                    q = policy.q_network.forward(s)
                    q_target_next = policy.target_network.forward(sn)
                    target = r + config.gamma * np.max(q_target_next) * (1 - d)

                    # TD update on output layer
                    lr = config.learning_rate
                    td_error = q[a] - target
                    h = np.maximum(0, s @ policy.q_network.W1 + policy.q_network.b1)
                    h = np.maximum(0, h @ policy.q_network.W2 + policy.q_network.b2)
                    policy.q_network.W_out[:, a] -= lr * td_error * h
                    policy.q_network.b_out[a] -= lr * td_error

                    # CQL: push down all Q-values slightly
                    cql_penalty = config.cql_alpha * 0.01
                    policy.q_network.b_out -= lr * cql_penalty

                    epoch_loss += td_error ** 2

        avg_loss = epoch_loss / max(1, len(buffer))
        losses.append(avg_loss)

        # Early stopping
        if epoch > 10 and len(losses) > 1:
            if abs(losses[-1] - losses[-2]) < 1e-6:
                break

    # Sync target network
    policy.sync_target_network()
    return losses
