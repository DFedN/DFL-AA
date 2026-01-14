
import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum


def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class MessageType(Enum):
    MODEL_UPDATE = "MODEL_UPDATE"
    COMPLETE = "COMPLETE"

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    num_classes: int
    channels: int
    img_size: int  # 28 for MNIST, 32 for CIFAR

MNIST_SPEC  = DatasetSpec("mnist",        (0.1307,),               (0.3081,),               10, 1, 28)
FMNIST_SPEC = DatasetSpec("fashion_mnist",(0.2860,),               (0.3530,),               10, 1, 28)
CIFAR10_SPEC= DatasetSpec("cifar10",      (0.4914,0.4822,0.4465),  (0.2023,0.1994,0.2010),  10, 3, 32)

# ==========================================
# Network Model (Distance-Based)
# ==========================================

@dataclass
class NetworkConditions:
    """Network conditions based on distance between nodes."""
    latency_ms: float        # one-way latency
    packet_loss: float       # per-chunk loss probability in [0,1]
    bandwidth_mbps: float    # Mbps


def calculate_network_conditions(
    distance_m: float,
    *,
    base_latency_ms: float = 1.0,
    latency_per_km: float = 3.0,
    base_loss: float = 0.01,
    loss_per_100m: float = 0.02,
    base_bandwidth_mbps: float = 10.0,
    out_of_range_distance: float = 500.0,
) -> Optional[NetworkConditions]:
    """
    Returns None if out of range.
    """
    if distance_m > out_of_range_distance:
        return None

    latency = base_latency_ms + (distance_m / 1000.0) * latency_per_km

    loss = base_loss + (distance_m / 100.0) * loss_per_100m
    loss = float(min(0.95, max(0.0, loss)))

    bandwidth_factor = 1.0 - (distance_m / out_of_range_distance)
    bandwidth = base_bandwidth_mbps * max(0.1, bandwidth_factor)

    return NetworkConditions(latency_ms=float(latency), packet_loss=loss, bandwidth_mbps=float(bandwidth))


# ==========================================
# Mobility Manager
# ==========================================

class MobilityManager:
    """Manages node positions over time from CSV."""

    def __init__(self, csv_path: str, num_nodes: int):
        df = pd.read_csv(csv_path)

        # Auto-detect columns
        def pick(cols: List[str]) -> str:
            for c in cols:
                if c in df.columns:
                    return c
            raise ValueError(f"Missing column from {cols}")

        tcol = pick(["time_sec", "time", "t"])
        ncol = pick(["node_id", "node", "id"])
        xcol = pick(["x_m", "x", "pos_x"])
        ycol = pick(["y_m", "y", "pos_y"])

        df = df[[tcol, ncol, xcol, ycol]].copy()
        df.columns = ["time", "node_id", "x", "y"]
        df = df.sort_values(["time", "node_id"])

        # Store positions: positions[time][node_id] = (x, y)
        self.positions: Dict[int, Dict[int, Tuple[float, float]]] = {}

        for t in df["time"].unique():
            self.positions[int(t)] = {}
            block = df[df["time"] == t]
            for _, row in block.iterrows():
                nid = int(row["node_id"])
                if 0 <= nid < num_nodes:
                    self.positions[int(t)][nid] = (float(row["x"]), float(row["y"]))

        self.times = sorted(self.positions.keys())
        self.num_nodes = num_nodes

    def get_positions(self, sim_time_sec: float) -> Dict[int, Tuple[float, float]]:
        """Get positions at given simulation time."""
        t = int(sim_time_sec)
        if t in self.positions:
            return self.positions[t]

        nearest = min(self.times, key=lambda x: abs(x - t))
        return self.positions[nearest]

    def get_distance(self, sim_time_sec: float, node_i: int, node_j: int) -> float:
        """Get distance between two nodes at given time."""
        pos = self.get_positions(sim_time_sec)
        if node_i not in pos or node_j not in pos:
            return float('inf')

        xi, yi = pos[node_i]
        xj, yj = pos[node_j]
        return float(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))


# ==========================================
# Model (MNIST)
# ==========================================

class MNISTMLP(nn.Module):
    """Simple fully connected network for MNIST"""

    def __init__(self,
                 input_dim: int = 28 * 28,
                 hidden_dims=(256, 128),
                 num_classes: int = 10,
                 dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

# ==========================================
# Model (Fashion-MNIST)
# ==========================================
class MNISTNet(nn.Module):
    """Improved CNN for MNIST"""

    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 3 * 3)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def resnet18_cifar(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    ResNet-18 adapted for CIFAR (32x32):
      - conv1: 3x3, stride=1, padding=1
      - remove maxpool
      - fc -> num_classes
    """
    # robust across torchvision versions
    try:
        # torchvision newer API
        if pretrained:
            from torchvision.models import ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = resnet18(weights=None)
    except Exception:
        # older API
        model = resnet18(pretrained=pretrained)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ==========================================
# Model Schema (Pre-Agreed - All Nodes Know This)
# ==========================================

class ModelSchema:
    """
    Pre-agreed model structure that all nodes know.
    In real FL, all nodes have the same architecture!
    No need to transmit this - it's known a priori.
    """

    def __init__(self, model: nn.Module):
        self.param_info = []  # (name, start_byte, end_byte, shape, dtype)
        byte_offset = 0

        for name, param in model.state_dict().items():
            arr = param.detach().cpu().numpy()
            param_bytes = arr.tobytes()

            self.param_info.append((
                name,
                byte_offset,
                byte_offset + len(param_bytes),
                tuple(arr.shape),
                arr.dtype.str,  #
            ))
            byte_offset += len(param_bytes)

        self.total_bytes = byte_offset

    def get_total_bytes(self) -> int:
        return self.total_bytes

    def get_param_info(self) -> List[Tuple]:
        return self.param_info


@torch.no_grad()
def model_to_bytes(model: nn.Module) -> bytes:
    """Convert model to raw bytes."""
    state_dict = model.state_dict()
    all_bytes = []

    for name, param in state_dict.items():
        param_bytes = param.cpu().numpy().tobytes()
        all_bytes.append(param_bytes)

    return b''.join(all_bytes)


@torch.no_grad()
def bytes_to_model(
        model_bytes: bytes,
        schema: ModelSchema,
        fill_missing: bool = False,
        local_model: Optional[nn.Module] = None
) -> Dict[str, torch.Tensor]:
    """
    Convert bytes back to model state dict using pre-agreed schema.

    Args:
        model_bytes: Received bytes (may have missing chunks = zero bytes)
        schema: Pre-agreed model structure
        fill_missing: If True and local_model provided, fill zero regions with local values
        local_model: Local model to use for filling missing parts
    """
    state_dict = {}

    # Convert to bytearray for easier manipulation
    full_bytes = bytearray(model_bytes)

    for name, start_byte, end_byte, shape, dtype_str in schema.get_param_info():
        param_bytes = bytes(full_bytes[start_byte:end_byte])

        # Check if this region is all zeros (missing chunks)
        is_missing = all(b == 0 for b in param_bytes)

        if is_missing and fill_missing and local_model is not None:
            # Use local model parameter
            local_state = local_model.state_dict()
            if name in local_state:
                param_tensor = local_state[name].cpu()
            else:
                # Fallback: create from zeros
                param_array = np.frombuffer(param_bytes, dtype=np.dtype(dtype_str))
                param_tensor = torch.from_numpy(param_array.copy()).reshape(shape)
        else:
            # Reconstruct from bytes
            param_array = np.frombuffer(param_bytes, dtype=np.dtype(dtype_str))
            param_tensor = torch.from_numpy(param_array.copy()).reshape(shape)

        state_dict[name] = param_tensor

    return state_dict


# ==========================================
# Message Types (Self-Contained UDP Packets)
# ==========================================
@dataclass
class BundledChunkUpdate:
    """
    One (sender -> receiver) message containing ONLY the chunks that survived loss.
    This preserves per-chunk loss semantics but avoids per-chunk Ray messages.
    """
    sender_id: str
    send_time: float
    total_chunks: int
    chunk_size: int          # bytes per chunk
    total_size: int          # total model bytes
    received_chunks: List[Tuple[int, bytes]]  # [(chunk_id, chunk_bytes), ...]
    latency: float
    pl: float
    round_number: int
    type_of_message: MessageType

    # Optional: Add checksum for integrity
    # checksum: int


    @property
    def received_chunk_ids(self) -> Set[int]:
        return {cid for cid, _ in self.received_chunks}

    @property
    def completeness(self) -> float:
        return len(self.received_chunks) / max(1, self.total_chunks)

    def to_bytes_zero_filled(self) -> bytes:
        """Reconstruct full model bytes with zeros for missing chunks."""
        out = bytearray(self.total_size)
        for cid, data in self.received_chunks:
            start = cid * self.chunk_size
            end = min(start + len(data), self.total_size)
            out[start:end] = data
        return bytes(out)

    def to_bytes_fill_local(self, local_bytes: bytes) -> bytes:
        """
        Fill missing with local model: start from local bytes, overwrite only received chunks.
        (This is the correct 'fill_local' semantics at chunk granularity.)
        """
        out = bytearray(local_bytes)  # already full length
        for cid, data in self.received_chunks:
            start = cid * self.chunk_size
            end = min(start + len(data), self.total_size)
            out[start:end] = data
        return bytes(out)



# ==========================================
# Federated Node
# ==========================================
@ray.remote
class DFedNode:
    """
    Federated Learning Node: A single DFL node running asynchronously.

    Each node:
    - Trains locally at its own pace
    - Broadcasts model chunks to neighbors
    - Receives partial updates (some chunks lost)
    - Aggregates using received chunks


    softSGD,softSGD_s, dflaa, dflaa_s
    """

    def __init__(
            self,
            node_id,
            train_data,
            train_labels,
            test_data,
            test_labels,
            num_nodes,

            hyperparams,
            total_train_data_length,
            client_data_info,
            dataset_name: str = "mnist",
            aggregation: str = "softSGD",
            mobility_csv: str = "mobility.csv",
    ):

        self.state_lock = asyncio.Lock()

        self.num_nodes = num_nodes
        self.node_id = node_id
        self.device = get_device()
        print(f"Node {node_id} device: {self.device.type}")

        self.aggregation = aggregation
        self.client_data_info = client_data_info
        self.total_train_data_length = total_train_data_length

        # Initialize dataset and model
        dataset_name_l = dataset_name.lower()
        if dataset_name_l in ["fashionmnist", "fashion_mnist", "fmnist"]:
            self.dataset = FMNIST_SPEC
            model = MNISTNet()
        elif dataset_name_l in ["cifar", "cifar10", "cifar-10"]:
            self.dataset = CIFAR10_SPEC
            model = resnet18_cifar(num_classes=10, pretrained=False)
            # model = build_resnet18_cifar(num_classes=self.dataset.num_classes)
        else:
            self.dataset = MNIST_SPEC
            model = MNISTMLP()

        # Schema MUST match the chosen model
        # dummy_model = type(model)() if not isinstance(model, nn.Sequential) else model  # safe-ish
        self.schema = ModelSchema(model)

        self.model = model.to(self.device)

        self.hyperparams = hyperparams
        if dataset_name_l in ["cifar", "cifar10", "cifar-10"]:
            # ResNet18-CIFAR: SGD recipe
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=hyperparams.get("learning_rate", 0.1),
                momentum=0.9,
                weight_decay=hyperparams.get("weight_decay", 5e-4),
                nesterov=True,
            )
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=hyperparams.get("milestones", [60, 120, 160]),
                gamma=hyperparams.get("gamma", 0.2),
            )
        else:
            # MNIST/FMNIST small nets: Adam is fine
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=hyperparams.get("learning_rate", 1e-3),
                weight_decay=hyperparams.get("weight_decay", 0.0),
            )
            self.scheduler = None

        self.criterion = nn.CrossEntropyLoss()

        # Training parameters
        self.local_epochs = hyperparams.get('local_epochs', 5)
        self.batch_size = hyperparams.get('batch_size', 64)

        # Create data loaders
        self.train_loader = self._create_data_loader(train_data, train_labels, shuffle=True)
        self.test_loader = self._create_data_loader(test_data, test_labels, shuffle=False)

        # Networking
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.runtime = hyperparams.get('runtime', 600)
        self.base_latency = hyperparams.get('base_latency_ms', 1.0)
        self.latency_per_km = hyperparams.get('latency_per_km', 3.0)
        self.base_loss = hyperparams.get('base_loss_ms', 0.02)
        self.loss_per_100m = hyperparams.get('loss_per_100m', 0.03)
        self.out_of_range = hyperparams.get('out_of_range', 500.0)
        self.chunk_size_kb = hyperparams.get('chunk_size_kb', 4)
        self.sim_start_time = None
        self.sim_end_time = None
        self.mobility = MobilityManager(mobility_csv, num_nodes)
        self.network_history : Dict[int, Dict[str, Dict[str, float]]] = {}

        # Model and neighbor data
        self.neighbor_models: Dict[str, Dict] = {}
        self.neighbor_round_counter: Dict[str, int] = {}
        self.neighbor_data_loss: Dict[str, int] = {}
        self.last_update: Dict[str, float] = {}
        self.global_start_time = None

        # Stats
        self.messages_received = 0
        self.messages_sent = 0
        self.total_chunks_sent = 0
        self.total_chunks_dropped = 0
        self.completed_nodes = []
        self.completed = False
        self.rng = np.random.default_rng(42)
        self.neighbor_completeness: Dict[str, float] = {}
        self.staleness: Dict[int, Dict[str, float]] = {}

        # Metrics
        self.round = 0
        self.accuracy_log_after = []
        self.loss_log_after = []
        self.round_history: Dict[int, Dict[str, Dict[str, float]]] = {}

        # Add for adaptive mixing
        self.theta_old = None
        self.alignment_threshold = 0.0  # Only use aligned neighbors

        # AoI
        self.last_gen_time = {}  # sender_id -> msg.send_time (global clock)
        self.last_arrival_time = {}  # sender_id -> arrival time at receiver (global clock)
        self.last_delay = {}  # sender_id -> (arrival - send_time)

        self.aoi_log = []
        self.aoi_round_log = []

    def _create_data_loader(self, data, labels, shuffle=True):
        # DATA -> torch float32
        if isinstance(data, np.ndarray):
            data = torch.tensor(data.copy(), dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            data = data.float()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Fix shapes
        # MNIST-like: (N,28,28) -> (N,1,28,28)
        if data.dim() == 3:
            data = data.unsqueeze(1)

        # CIFAR could be (N,32,32,3)
        if data.dim() == 4:
            if data.shape[1] in (1, 3):
                pass  # already NCHW
            elif data.shape[-1] in (1, 3):
                data = data.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
            else:
                raise ValueError(f"Unsupported 4D data shape: {tuple(data.shape)}")
        else:
            raise ValueError(f"Unsupported data dim: {data.dim()}, shape={tuple(data.shape)}")

        # Scale if [0..255]
        if float(data.max()) > 1.0:
            data = data / 255.0

        # Normalize (channel-wise)
        mean = torch.tensor(self.dataset.mean, dtype=torch.float32).view(1, self.dataset.channels, 1, 1)
        std = torch.tensor(self.dataset.std, dtype=torch.float32).view(1, self.dataset.channels, 1, 1)
        data = (data - mean) / std

        # LABELS
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels.copy(), dtype=torch.long)
        elif isinstance(labels, torch.Tensor):
            labels = labels.long()
        else:
            raise TypeError(f"Unsupported labels type: {type(labels)}")

        dataset = TensorDataset(data, labels)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )

    def get_accuracy_log_after_aggregation(self) -> List[Tuple[int, float]]:
        return self.accuracy_log_after

    def get_loss_log_after_aggregation(self) -> List[Tuple[int, float]]:
        return self.loss_log_after

    def get_round_history(self) -> Dict[int, Dict[str, dict[str, float]]]:
        return self.round_history

    def get_start_time(self):
        return self.sim_start_time

    def get_end_time(self):
        return self.sim_end_time

    def get_network_history(self) -> Dict[int, Dict[str, dict[str, float]]]:
        return self.network_history

    def get_staleness(self):
        return self.staleness

    def get_current_time(self) -> float:
        """Get current simulation time."""
        return time.time() - self.sim_start_time if self.sim_start_time else 0.0

    def get_aoi_log(self):
        return self.aoi_log

    def get_aoi_round_log(self):
        return self.aoi_round_log

    def get_global_time(self) -> float:
        """Current time in global synchronized clock"""
        if self.global_start_time is None:
            raise RuntimeError("Global start time not set!")
        return time.time() - self.global_start_time

    async def start(self, global_start_time: float):
        """Start the federated node"""
        self.sim_start_time = time.time()
        self.global_start_time = global_start_time

        asyncio.create_task(self.process_messages())

        await asyncio.sleep(0.5)

        print(f"Starting federated Node {self.node_id} at {self.sim_start_time}")
        await self.training_loop()

    async def receive_message(self, msg: BundledChunkUpdate):
        """Receive message from neighbor"""
        self.messages_received += 1
        now = self.get_global_time()

        # time_diff = self.get_current_time() - msg.send_time
        time_diff = now - msg.send_time
        if time_diff <= msg.latency:
            await asyncio.sleep(msg.latency - time_diff)

        sid = msg.sender_id
        self.last_gen_time[sid] = float(msg.send_time)
        self.last_arrival_time[sid] = float(now)
        self.last_delay[sid] = float(now - msg.send_time)
        await self.message_queue.put(msg)

    async def local_training(self):
        """Perform local training"""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if self.scheduler is not None:
                self.scheduler.step()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"[Node {self.node_id}] Local training complete. Avg loss: {avg_loss:.4f}")

    def evaluate_model(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate model on test set"""
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion(output, target)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                loss_sum += loss.item() * data.size(0)

        accuracy = correct / total if total > 0 else 0
        avg_loss = loss_sum / total if total > 0 else 0
        return accuracy, avg_loss

    async def training_loop(self):
        """
        NEW: Polished training loop
        """
        try:
            while True:
                current_time = time.time() - self.sim_start_time
                if self.get_global_time() >= self.runtime:
                    break

                # 1) start round
                self.round += 1

                # 2) local training on local data
                await self.local_training()

                # 3) evaluate after training
                accuracy, loss = self.evaluate_model(self.model)
                self.accuracy_log_after.append((self.round, self.get_global_time(), accuracy * 100))
                self.loss_log_after.append((self.round, self.get_global_time(), loss))
                print(f"[Node{self.node_id}] Round {self.round} - Test Accuracy: {accuracy * 100:.2f}%  Test Loss; {loss:.2f}")

                # 4) Send updated model to neighbors in both active and inactive cases
                # This is to ensure unblocking in changing network
                await self.send_model_update()

                self._log_aoi_stats()

                # 5) Aggregate with available neighbors
                # ===== USE UNIFIED AGGREGATION METHOD =====
                if self.aggregation in ("dflaa", "dflaa_s", "softSGD", "softSGD_s", "dflaa_c", "softSGD_c"):
                    aggregated_state = await self.aggregate_with_ablation()
                else:
                    # Fallback to original methods
                    if self.aggregation == "fedavg":
                        aggregated_state = await self.fedavg_aggregate()
                    else:
                        aggregated_state = self.model.state_dict()

                # 6) Set aggregated parameters
                self.model.load_state_dict(aggregated_state)

                await asyncio.sleep(0.1)

            # Training complete
            if not self.completed:
                await self.send_complete()
                self.completed = True
                self.sim_end_time = time.time()
                print(f"[Node {self.node_id}] Training completed at round {self.round}")

        except Exception as e:
            print(f"ERROR [Node {self.node_id}] Training loop: {e}")
            raise

    async def aggregate_with_ablation(self):
        """
        Unified aggregation method supporting all ablation variants:
        - dflaa: Full DFLAA (delta + completeness + staleness)
        - dflaa_s: Delta only (no weighting)
        - softSGD: Soft-DSGD (fill-local + FedAvg)
        - softSGD_s: Soft-DSGD + staleness/completeness weighting
        """
        async with self.state_lock:
            local = self.model.state_dict()

            if not self.neighbor_models:
                return local

            now = self.get_global_time()

            # Hyperparameters
            tau = float(self.hyperparams.get("staleness_tau_sec", 1.0))
            min_comp = float(self.hyperparams.get("min_completeness", 0.2))

            # Identify float tensors (exclude BN running stats)
            float_keys = []
            for k, v in local.items():
                if not torch.is_tensor(v):
                    continue
                if not (v.is_floating_point() or v.is_complex()):
                    continue
                # Exclude BN running statistics
                if 'running_mean' in k or 'running_var' in k:
                    continue
                float_keys.append(k)

            # =============================================
            # Compute weights and decide aggregation style
            # =============================================

            weights = {}  # neighbor_id -> weight
            for nid, neigh in self.neighbor_models.items():
                comp = 1.0 - float(self.neighbor_data_loss.get(nid, 1.0))

                # Apply completeness filtering for weighted variants
                if self.aggregation in ("dflaa", "softSGD_s", "dflaa_c", "softSGD_c"):
                    if comp < min_comp:
                        continue


                # 2) Assign weight for each variant (ALWAYS assign w!)
                if self.aggregation in ("dflaa_c", "softSGD_c"):
                    w = comp
                elif self.aggregation in ("dflaa", "softSGD_s"):
                    recv_t = float(self.last_update.get(nid, 0.0))
                    stale_sec = max(0.0, now - recv_t)
                    st = float(np.exp(-stale_sec / max(1e-6, tau)))
                    w = comp * st
                else:
                    w = 1.0

                if w > 0.0:
                    weights[nid] = w

            if not weights:
                print(f"[Node {self.node_id}] No valid neighbors for aggregation")
                return local

            # =============================================
            # Choose aggregation style: DELTA vs FedAvg
            # =============================================

            if self.aggregation in ("dflaa", "dflaa_s", "dflaa_c"):
                # ===== DELTA AGGREGATION =====
                sum_w = sum(weights.values())
                delta_sum = {k: torch.zeros_like(local[k]) for k in float_keys}

                for nid, neigh in self.neighbor_models.items():
                    if nid not in weights:
                        continue

                    w = weights[nid]
                    for k in float_keys:
                        nk = neigh[k]
                        lk = local[k]
                        # Type/device safety
                        if nk.dtype != lk.dtype:
                            nk = nk.to(dtype=lk.dtype)
                        if nk.device != lk.device:
                            nk = nk.to(device=lk.device)

                        delta_sum[k] += (w * (nk - lk))

                denom = 1.0 + sum_w
                out = dict(local)
                for k in float_keys:
                    out[k] = local[k] + (delta_sum[k] / denom)

                print(f"[Node {self.node_id}] {self.aggregation}: "
                  f"neighbors={len(weights)}, sum_w={sum_w:.3f}")
                return out

            else:
                # ===== FedAvg AGGREGATION (Soft-DSGD style) =====
                sum_w = sum(weights.values())

                # Initialize aggregated state
                aggregated = {}

                # Copy non-float tensors from local
                other_keys = [k for k in local.keys() if k not in float_keys]
                for k in other_keys:
                    v = local[k]
                    aggregated[k] = v.clone() if torch.is_tensor(v) else v

                # Initialize float tensors
                for k in float_keys:
                    aggregated[k] = torch.zeros_like(local[k])

                # Compute own weight
                own_weight = 1.0 / (1.0 + sum_w)

                # Add weighted neighbors
                for nid, neigh in self.neighbor_models.items():
                    if nid not in weights:
                        continue

                    w = weights[nid] / (1.0 + sum_w)
                    for k in float_keys:
                        nk = neigh[k]
                        lk = local[k]
                        if nk.device != lk.device:
                            nk = nk.to(device=lk.device)
                        if nk.dtype != lk.dtype:
                            nk = nk.to(dtype=lk.dtype)

                        aggregated[k].add_(nk, alpha=w)

                # Add own model
                for k in float_keys:
                    aggregated[k].add_(local[k], alpha=own_weight)

                print(f"[Node {self.node_id}] {self.aggregation}: "
                  f"neighbors={len(weights)}, sum_w={sum_w:.3f}")
                return aggregated

    async def fedavg_aggregate(self):
        """
        FedAvg aggregation with dtype safety:
          - Aggregates ONLY floating/complex tensors.
          - Copies non-float tensors (e.g., BN num_batches_tracked: Long) from local.
          - Casts neighbor tensors to local dtype/device for float keys.
          - Logs per-round neighbor info like your original.
        """
        async with self.state_lock:
            # ---- choose valid neighbors (keep your logic) ----
            if self.aggregation in ("vanilla", "fedavg"):
                valid_neighbors = [
                    nid for nid in self.neighbor_models.keys()
                    if float(self.neighbor_data_loss.get(nid, 1.0)) <= 0.08
                ]
            elif self.aggregation == "softSGD" or self.aggregation == "softSGD_s":
                valid_neighbors = list(self.neighbor_models.keys())
            else:
                valid_neighbors = list(self.neighbor_models.keys())

            # Hyperparams (safe defaults)
            tau = float(self.hyperparams.get("staleness_tau_sec", 30.0))
            min_comp = float(self.hyperparams.get("min_completeness", 0.2))
            now = self.get_global_time()

            if not valid_neighbors:
                print(f"WARNING [Node {self.node_id}] Round {self.round}: No neighbor models for aggregation.")
                return self.model.state_dict()

            local_state = self.model.state_dict()

            # ---- keys to aggregate safely ----
            float_keys = [
                k for k, v in local_state.items()
                if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex())
            ]
            other_keys = [k for k in local_state.keys() if k not in float_keys]

            # ---- weights ----
            weights = {}
            total_weights = 0
            for nid, neigh in self.neighbor_models.items():
                comp = 1.0 - float(self.neighbor_data_loss.get(nid, 1.0))
                if comp < min_comp:
                    continue

                recv_t = float(self.last_update.get(nid, 0.0))
                stale_sec = max(0.0, now - recv_t)

                st = float(np.exp(-stale_sec / max(1e-6, tau)))
                w = comp * st

                weights[nid] = w
                total_weights += w

            weights[self.node_id] = 1
            total_weights += weights[self.node_id]

            if total_weights <= 0.0:
                # fallback: only self
                weights = {self.node_id: 1.0}
                total_weights = 1.0

            norm_weights = {nid: w / total_weights for nid, w in weights.items()}


            denom = float(len(valid_neighbors) + 1)
            own_weight = 1.0 / denom
            neigh_weight = 1.0 / denom  # equal weights per neighbor (your current behavior)

            # ---- init aggregated dict ----
            aggregated = {}

            # copy non-float tensors EXACTLY from local (BN counters, masks, etc.)
            for k in other_keys:
                v = local_state[k]
                aggregated[k] = v.clone() if torch.is_tensor(v) else v

            # init float accumulators as zeros
            for k in float_keys:
                aggregated[k] = torch.zeros_like(local_state[k])

            # ---- accumulate neighbors (float keys only) ----
            for nid in valid_neighbors:
                neigh = self.neighbor_models[nid]
                for k in float_keys:
                    nk = neigh[k]
                    lk = local_state[k]
                    # ensure device + dtype match local
                    if nk.device != lk.device:
                        nk = nk.to(device=lk.device)
                    if nk.dtype != lk.dtype:
                        nk = nk.to(dtype=lk.dtype)
                    # in-place add with alpha avoids extra casts
                    aggregated[k].add_(nk, alpha=norm_weights[nid])

            # ---- add own model (float keys only) ----
            for k in float_keys:
                aggregated[k].add_(local_state[k], alpha=norm_weights[self.node_id])

            # ---- round history logging (same idea as before) ----
            round_info = {}
            for nid in valid_neighbors:
                round_info[nid] = {
                    "neighbor_round": int(self.neighbor_round_counter.get(nid, 0)),
                    "own_round": int(self.round),
                    "weight": float(neigh_weight),
                    "pkt_loss": float(self.neighbor_data_loss.get(nid, 0.0)),
                }
            self.round_history[int(self.round)] = round_info

            print(f"[Node {self.node_id}] FedAvg-safe aggregation complete: neighbors={len(valid_neighbors)}")
            return aggregated

    async def process_messages(self):
        """Background message processor"""
        while not self.completed:
            try:
                msg = await self.message_queue.get()
                print(f"[Node {self.node_id}] Received message from neighbor: Node {msg.sender_id} at round: {self.round}, and their round: {msg.round_number}")

                if msg.type_of_message == MessageType.MODEL_UPDATE:
                    await self.handle_model_update(msg)
                elif msg.type_of_message == MessageType.COMPLETE:
                    await self.handle_complete(msg)

            except Exception as e:
                print(f"ERROR [Node {self.node_id}] Error processing message: {e}")

    async def handle_model_update(self, msg):
        """Handle model update message"""
        sender_id = msg.sender_id

        async with self.state_lock:
            local_bytes = model_to_bytes(self.model)

            if self.aggregation in ("softSGD", "dflaa", "softSGD_s", "dflaa_s"):
                neigh_model_bytes = msg.to_bytes_fill_local(local_bytes)
            else:
                neigh_model_bytes = msg.to_bytes_zero_filled()

            recon_state = bytes_to_model(neigh_model_bytes, self.schema, fill_missing=False, local_model=None)
            # IMPORTANT: make sure neighbor tensors are on same device as local
            recon_state = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in recon_state.items()
            }

            self.neighbor_models[sender_id] = recon_state
            self.neighbor_round_counter[sender_id] = msg.round_number
            self.neighbor_data_loss[sender_id] = msg.pl
            self.last_update[sender_id] = msg.send_time
            self.neighbor_completeness[sender_id] = float(msg.completeness)

            print(f"[Node {self.node_id}] Received round {msg.round_number} update from Node {sender_id}")

    async def handle_complete(self, msg):
        """Handle completion message"""
        sender_id = msg.sender_id

        async with self.state_lock:
            local_bytes = model_to_bytes(self.model)

            if self.aggregation in ("softSGD", "dflaa", "softSGD_s", "dflaa_s"):
                neigh_model_bytes = msg.to_bytes_fill_local(local_bytes)
            else:
                neigh_model_bytes = msg.to_bytes_zero_filled()

            recon_state = bytes_to_model(neigh_model_bytes, self.schema, fill_missing=False, local_model=None)
            recon_state = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in recon_state.items()
            }

            self.neighbor_models[sender_id] = recon_state
            self.neighbor_round_counter[sender_id] = msg.round_number
            self.neighbor_data_loss[sender_id] = msg.pl
            self.last_update[sender_id] = msg.send_time
            self.neighbor_completeness[sender_id] = float(msg.completeness)

            if sender_id not in self.completed_nodes:
                self.completed_nodes.append(sender_id)

            print(f"[Node {self.node_id}] {sender_id} completed training")

    async def send_model_update(self):
        """Send model update to neighbors"""
        total_msgs = await self.broadcast_model(model_to_bytes(self.model), MessageType.MODEL_UPDATE)
        if not total_msgs:
            return
        for neighbor_name, msg in total_msgs.items():
            try:
                if neighbor_name == self.node_id:
                    continue
                if neighbor_name in self.completed_nodes:
                    continue

                neighbor = ray.get_actor(neighbor_name)
                neighbor.receive_message.remote(msg)
                self.messages_sent += 1
                print(f"[Node {self.node_id}] Sent round {self.round} update to Node {neighbor_name}")
            except Exception as e:
                print(f"WARNING [Node {self.node_id}] Could not send to Node {neighbor_name}: {e}")

    async def send_complete(self):
        """Send completion message to neighbors"""
        total_msgs = await self.broadcast_model(model_to_bytes(self.model), MessageType.COMPLETE)
        if not total_msgs:
            return
        for neighbor_name, msg in total_msgs.items():
            try:
                if neighbor_name == self.node_id:
                    continue
                if neighbor_name in self.completed_nodes:
                    continue

                neighbor = ray.get_actor(neighbor_name)
                neighbor.receive_message.remote(msg)
                self.messages_sent += 1
                print(f"[Node {self.node_id}] Sent COMPLETE to Node {neighbor_name}")
            except Exception as e:
                print(f"WARNING [Node {self.node_id}] Could not send COMPLETE to Node {neighbor_name}: {e}")

    async def broadcast_model(
            self,
            model_bytes: bytes,
            msg_type: MessageType,
    ) -> Dict[str, BundledChunkUpdate] | None:
        """
        Simulate broadcasting model chunks with PER-CHUNK packet loss.
        Each chunk can be lost independently!
        """
        total_msgs: Dict[str, BundledChunkUpdate] = {}
        current_time = self.get_current_time()
        # Current global time (when THIS node is sending)
        send_time_global = self.get_global_time()
        sender_pos_dict = self.mobility.get_positions(send_time_global)
        self_id = int(self.node_id.split("_")[-1])

        if self_id not in sender_pos_dict:
            print(f"ERROR [Node {self.node_id}] Node is not in position dict")
            return None

        # Split model into chunks
        total_size = len(model_bytes)
        chunk_size = self.chunk_size_kb * 1024
        num_chunks = (total_size + chunk_size - 1) // chunk_size

        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)
            chunk_data = model_bytes[start:end]
            chunks.append((i, chunk_data))

        # Send to each neighbor
        network_info = {}
        for receiver_id in range(self.num_nodes):
            if receiver_id == self_id:
                continue

            distance = self.mobility.get_distance(send_time_global, self_id, receiver_id)

            network_cond = calculate_network_conditions(
                distance,
                base_latency_ms=self.base_latency,
                latency_per_km=self.latency_per_km,
                base_loss=self.base_loss,
                loss_per_100m=self.loss_per_100m,
                out_of_range_distance=self.out_of_range,
            )

            if network_cond is None:
                continue  # Out of range

            received_chunks: List[Tuple[int, bytes]] = []
            received_bytes = 0

            for chunk_id, chunk_data in chunks:
                self.total_chunks_sent += 1

                # Losses are grater than 8% are considered as complete drops
                if network_cond.packet_loss > 0.08:
                    # Each chunk has independent packet loss probability
                    if self.rng.random() >= network_cond.packet_loss:
                        received_chunks.append((chunk_id, chunk_data))
                        received_bytes += len(chunk_data)
                    else:
                        self.total_chunks_dropped += 1
                else:
                    received_chunks.append((chunk_id, chunk_data))
                    received_bytes += len(chunk_data)

            # If nothing survived, do not send anything
            if not received_chunks:
                continue

            print(f"Transmitting model to Node {receiver_id}: {received_bytes} / {total_size} bytes")

            # Calculate transmission delay
            # Only received chunks contribute to transmission time
            # received_size_mb = (len(received_chunk_ids) * self.network_config.get("chunk_size_kb", 4)) / 1024.0
            # transmission_time_sec = received_size_mb / network_cond.bandwidth_mbps
            # total_delay_sec = (network_cond.latency_ms / 1000.0) + transmission_time_sec

            total_delay_sec = (network_cond.latency_ms / 1000.0)

            update = BundledChunkUpdate(
                sender_id=self.node_id,
                send_time=send_time_global,
                total_chunks=num_chunks,
                chunk_size=chunk_size,
                total_size=total_size,
                received_chunks=received_chunks,
                latency=total_delay_sec,
                pl=network_cond.packet_loss,
                round_number=self.round,
                type_of_message=msg_type,
            )

            receiver_id = f"node_{receiver_id}"
            network_info[receiver_id] = {
                "data_loss": network_cond.packet_loss,
                "latency": total_delay_sec,
                "total_chunks": num_chunks,
                "chunk_size": chunk_size,
                "total_size": total_size,
                "received_chunks": len(received_chunks),
            }

            total_msgs[receiver_id] = update

        # collect network history data
        self.network_history[self.round] = network_info

        return total_msgs

    def get_detailed_status(self):
        """Get detailed node status for monitoring"""
        acc, loss = self.evaluate_model(self.model)

        neighbors = []
        for nid, model in self.neighbor_models.items():
            neighbors.append({
                "id": nid,
                "last_round": self.neighbor_round_counter.get(nid, 0),
                "data_loss": self.neighbor_data_loss.get(nid, 0),
            })

        return {
            "node_id": self.node_id,
            "time_seconds": self.get_global_time(),
            "accuracy": acc * 100,
            "loss": loss,
            "round_counter": self.round,
            "completed": self.completed,
            "neighbors": neighbors,
            "neighbor_models_count": len(self.neighbor_models),
            "network_history": self.network_history,
        }


    def _log_aoi_stats(self):
        now = self.get_global_time()
        nbrs = list(self.neighbor_models.keys())

        aoi = []
        delay = []

        for nid in nbrs:
            if nid in self.last_gen_time:
                aoi.append(now - self.last_gen_time[nid])
            if nid in self.last_delay:
                delay.append(self.last_delay[nid])

        if not aoi:
            return

        row = (
            int(self.round),
            float(now),
            float(np.mean(aoi)),
            float(np.median(aoi)),
            float(np.percentile(aoi, 90)),
            float(np.max(aoi)),
            float(np.median(delay)) if delay else float("nan"),
            float(np.max(delay)) if delay else float("nan"),
        )
        self.aoi_log.append(row)

        # optional print every 50 rounds
        if self.round % 50 == 0:
            print(f"[Node {self.node_id}] AoI mean/med/p90/max = "
                  f"{row[2]:.2f}/{row[3]:.2f}/{row[4]:.2f}/{row[5]:.2f} sec | "
                  f"Delay med/max = {row[6]:.3f}/{row[7]:.3f} sec")
