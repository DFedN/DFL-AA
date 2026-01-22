# DFL-AA: Delta Aggregation for Mobile Decentralized Federated Learning over Lossy Links

Welcome to the code base for DFL-AA Simulator...

To get start with, install all the requirements listed down in the `requirements.txt`. Then follow each step to generate `1. Mobility Traces for given number of nodes`, `2. Split down data set among participating nodes`, `3. Run base experiments setup`, `4. Test for other scenarios`, and finaly generate the illustrations from the results. Please make sure to use exact same locations used in the code or change as you wanted with extra care !!!!

### Generate Mobility Traces for Given Nodes, Given area and Given Timespan: 

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

### Create a data partitioning according to the nodes

``` bash
python data_loader.py \
    --datasets mnist,fmnist,cifar10 \
    --alphas 0.1,0.5,1.0 \
    --num-clients 20 \
    --split non_iid_dirichlet \
    --out-root dataset_partitions \
    --seed 42
```


### Script for core experiments for given datasets + algorithm

``` bash
python run_all_exp.py \
    --partitions-root dataset_partitions \
    --partition-pattern "{root}/{dataset}/alpha_{alpha_tag}" \
    --datasets mnist,fmnist,cifar10 \
    --alphas 0.1,0.5,1.0 \
    --aggregations fedavg,softSGD,dflaa
```

test for scalability levels
python run_all_exp_scalable.py \
    --partitions-root dataset_partitions \
    --partition-pattern "{root}/{dataset}/alpha_{alpha_tag}" \
    --datasets mnist \
    --alphas 0.1,0.5 \
    --num-nodes 40 \
    --aggregations fedavg,softSGD,dflaa


test for ablation
python run_all_exp_abl.py \
    --partitions-root dataset_partitions \
    --partition-pattern "{root}/{dataset}/alpha_{alpha_tag}" \
    --datasets mnist \
    --alphas 0.1,0.5 \
    --aggregations softSGD_s,softGSD_c,dflaa_s,dflaa_c


test for different loss levels
python run_all_exp_loss.py \
    --partitions-root dataset_partitions \
    --partition-pattern "{root}/{dataset}/alpha_{alpha_tag}" \
    --datasets mnist \
    --alphas 0.1,0.5 \
    --aggregations fedavg,softSGD,dflaa


test for different tau levels
python run_all_exp_staledecay.py \
    --partitions-root dataset_partitions \
    --partition-pattern "{root}/{dataset}/alpha_{alpha_tag}" \
    --datasets mnist \
    --alphas 0.1,0.5 \
    --aggregations fedavg,softSGD,dflaa
