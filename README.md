# DFL-AA: Delta Aggregation for Mobile Decentralized Federated Learning over Lossy Links

Welcome to the **DFL-AA Simulator** codebase.

## DFL-AA in a nutshell

**DFL-AA (Delta-based Decentralized Federated Learning with Adaptive Aggregation)** is a lightweight framework designed for **mobile decentralized FL** where communication is **unreliable** (packet loss) and the network topology changes **rapidly** (high mobility). In such settings, classic methods break down: **FedAvg** wastes bandwidth by discarding partial updates, and **Soft-DSGD** can create **artificial consensus** by filling missing parameters with the local model—often hurting accuracy under severe non-IID data.

### Key idea: partial + stale updates done right
DFL-AA handles **partial** and **stale** neighbor updates using two simple mechanisms:

- **Delta Aggregation (avoids artificial consensus):** aggregate only the *received* parts of a neighbor model by using deltas, treating missing chunks as **Δ = 0 (neutral)** rather than “agreeing” with the local model.
- **Age-of-Information (AoI) Decay (true staleness):** weight updates by **wall-clock freshness** (AoI) instead of misleading round counters under heterogeneous speeds.

A practical weighting rule:
```text
weight = completeness * exp(-AoI / τ)
w_new  = w_local + Σ(weight * Δ) / (1 + Σ weight)
```

## System Diagram

<p align="center">
  <img src="https://github.com/DFedN/DFL-AA/blob/main/dflaa_overview.png" alt="DFL-AA System Diagram" width="85%">
</p>

##### Here is the promising performances of our method compared to other baselines (1. on MNIST and 2. on Fashion MNIST) 

<p align="center">
  <img src="https://github.com/DFedN/DFL-AA/blob/main/icdcs_paper_results/mnist_time_mean_1x3_zoom.png" alt="DFL-AA System Diagram" width="85%">
</p>

<p align="center">
  <img src="https://github.com/DFedN/DFL-AA/blob/main/icdcs_paper_results/fmnist_time_mean_1x3_zoom.png" alt="DFL-AA System Diagram" width="85%">
</p>

<br>
<br>

* * *
* * *
* * *

<br>

This repository provides the simulator code and scripts required to reproduce the experiments and generate figures reported in our work. The typical workflow is:

1. Generate **mobility traces** for a given number of nodes  
2. Create **data partitions** across participating nodes  
3. Run the **core experiments**  
4. Evaluate additional scenarios (scalability, packet loss, different τ settings, etc.)  
5. Generate **tables and plots** from the results  

> **Important:** The scripts assume specific default folder structures and output paths.  
> Please use the same locations as in the code, or update paths carefully if you change them.

<br>


## Installation

Install dependencies from:

```bash
pip install -r requirements.txt
```




### 1. Generate Mobility Traces

Generate mobility traces for a given number of nodes, area, and duration:

``` bash
python road_flow_ou_switch_mobility.py \
    --nodes 20 --area_m 2000 --duration_s 3000 --dt_s 1 \
    --num_flows 4 --flow_heading_mode paired --flow_heading_deg 0 --flow_heading_spread_deg 90 \
    --flow_speed_kmph 35 --flow_speed_spread_kmph 8 \
    --tau_rel_s 600 --sigma_rel_mps 0.25 \
    --tau_switch_s 900 --switch_mode adjacent \
    --speed_scale 0.7 --boundary reflect --out_csv mobility.csv \
    --analyze_radius_m 400 --analyze_delta_s 60
```

#### Optional: Generate a GIF (visualize node mobility)

If you want to generate a GIF showing node behaviors based on the mobility traces:

``` bash
python mobility_to_graph.py \
    --csv_path mobility.csv \
    --radius_m 500 \
    --area_m 2000 \
    --fps 10 \
    --out_gif network.gif
```

### 2. Create Dataset Partitions

Create dataset partitions across nodes (non-IID Dirichlet split by default):

``` bash
python data_loader.py \
    --datasets mnist,fmnist,cifar10 \
    --alphas 0.1,0.5,1.0 \
    --num-clients 20 \
    --split non_iid_dirichlet \
    --out-root dataset_partitions \
    --seed 42
```


### 3. Run Core Experiments

Run the main experiment suite for datasets and aggregation methods. Results will be saved to the configured output location.

``` bash
python run_all_exp.py \
    --partitions-root dataset_partitions \
    --partition-pattern "{root}/{dataset}/alpha_{alpha_tag}" \
    --datasets mnist,fmnist,cifar10 \
    --alphas 0.1,0.5,1.0 \
    --aggregations fedavg,softSGD,dflaa
```

#### Notes

* **Scalability tests**: increase the node count in relevant scripts/commands and rerun the same pipeline.
Adjust compute resources appropriately for your machine.
* **Packet loss evaluation**: modify the network initialization parameters in dfl_core.py, then rerun the experiments.
* **Different `tau` ranges**: update the `tau` values in `run_all_exp.py`, then rerun (typically using only dflaa for aggregation).


### 4. Component-wise / Ablation Evaluation (DFL-AA)

Run component-wise evaluation for DFL-AA using specific aggregation variants:

``` bash
python run_all_exp_abl.py \
    --partitions-root dataset_partitions \
    --partition-pattern "{root}/{dataset}/alpha_{alpha_tag}" \
    --datasets mnist \
    --alphas 0.1,0.5 \
    --aggregations softSGD_s,softGSD_c,dflaa_s,dflaa_c
```




### Generate Tables and Figures

#### (a) Main results

```bash
python main_results.py \
    --results-root all_exp_results \
    --datasets mnist,fmnist \
    --alphas 1.0,0.5,0.1 \
    --aggregations fedavg,softSGD,dflaa \
    --out-dir icdcs_paper_results \
    --tables-out ICDCS_tables.txt
```

#### (b) Ablation / component-wise results

``` bash
python main_results_comp_abl.py \
    --root ablation_results \
    --dataset mnist \
    --alpha 0.1,0.5 \
    --aggregations dflaa,dflaa_s,dflaa_c,softSGD,softGSD_c \
    --out-dir icdcs_paper_results
```

> **Tip:** For results generated under custom modifications (e.g., altered core parameters), reuse the logic in `main_results.py` and pass the correct results directory.


<br>

* * *

#### Anonymity Note

*This repository was created **solely for anonymous sharing** of the simulator code and results for **reproducibility**. It was created **after all paper experiments were completed**, and the code/results were then **copied and organized** here.*

*As a result, this repository **does not include the full development history or commit record** from the original private repository. However, it contains **all simulator code used to produce the reported results**—only the full commit history is missing.*





