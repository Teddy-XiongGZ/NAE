# Neural Additive Experts: Context-Gated Experts for Controllable Model Additivity

## Dataset Setup

The following links are shared by the authors of NODE-GAM (Chang et al., 2022).

1. California Housing
- Since we follow NBM (Radenovic et al., 2022) to use the sklearn api for housing dataset, no files need to be downloaded in advance.

2. MIMIC-II
- Move to the `data` directory and create the `mimic2` directory under it
    - `mv ./data`
    - `mkdir mimic2`
- Download the files from the following link
    - https://1drv.ms/u/s!ArHmmFHCSXTIg8dnoN2EHpDJiNWkjg?e=qdWLMK
- Unzip the downloaded file
    - `unzip *.zip`

3. MIMIC-III
- Move to the `data` directory and create the `mimic3` directory under it
    - `mv ./data`
    - `mkdir mimic3`
- Download the files from the following link
    - https://1drv.ms/u/s!ArHmmFHCSXTIg8d6o6icAULya24iyw?e=s7TNxa
- Unzip the downloaded file
    - `unzip *.zip`

4. Adult Income
- Move to the `data` directory and create the `income` directory under it
    - `mv ./data`
    - `mkdir income`
- Download the files from the following link
    - https://1drv.ms/u/s!ArHmmFHCSXTIg8d9xcRG4wxNyU2C6w?e=XCb4QX
- Unzip the downloaded file
    - `unzip *.zip`

5. Credit Income
- Move to the `data` directory and create the `credit` directory under it
    - `mv ./data`
    - `mkdir credit`
- Download the files from the following link
    - https://1drv.ms/u/s!ArHmmFHCSXTIg8d_VZKCFHe7_uNkpw?e=vOSV3S
- Unzip the downloaded file
    - `unzip *.zip`

6. Year
- Move to the `data` directory and create the `year` directory under it
    - `mv ./data`
    - `mkdir year`
- Download the files from the following link
    - https://www.dropbox.com/s/l09pug0ywaqsy0e/YearPredictionMSD.txt?dl=1
- Unzip the downloaded file
    - `unzip *.zip`

## Usage

The running scripts are included in the `./scripts` directory. Here we give an example to run NAE on the Housing dataset
```bash
for seed in `seq 0 9`
do
    python src/train.py --device cuda:0 --seed $seed --exp_str optim_$seed --model NAE --data housing
done
```
