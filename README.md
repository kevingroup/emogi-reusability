# emogi-reusability
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


#### Cancer gene prediction with co-expression network

To reproduce the results for cancer gene prediction using co-expression network, you should install the EMOGI model based on the original document (https://github.com/schulter/EMOGI). Then you can use the following parameters for training

```bash
python train_EMOGI_cv.py -e 5000 -s 1 -wd 0.005 -hd 20 40 -lm 90 -cv 10 -seed ${seed} -d ./co-expression/CPDB_coexp_multiomics.h5
```

#### Essential gene prediction 

The input h5 data files for the essential gene prediction can be found in the `essential_gene` folder. The following command can be used to reproduce our results

```bash
python train_EMOGI_cv.py -e 5000 -s 1 -wd 0.005 -hd 300 100 -lm 5 -cv 10 -seed ${seed} -d ./essential_gene/CPDB_essential_multiomics.h5
```

#### Cancer gene prediction with Graph Attention Network (GAT)

Here we provided two versions of GAT (one based on the original tensorflow GAT library and the other based on pytorch DGL library).

For the tensorflow implementation from the original GAT paper, you could follow the document to install the required packages (https://github.com/PetarV-/GAT). And then you also need to install the other required packages with `pip install h5py numpy scipy`

After installing the packages, you can use the command below to train the GAT models

```bash
python train_GAT.py -e 5000 -lr 0.01 -lm 90 -hd 8 -ah 8 1  -do 0.25 -wd 0.0 -seed 1 -d ${PATH_TO_INPUT_H5_FILE}
```

For the DGL version, to make it consistent with the tensorflow version here we also used cuda 10.2. You need to first install the pytorch library with `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch` as well as the dgl libray with `conda install -c dglteam dgl-cuda10.2` . Then the other required packages can be installed with `pip install h5py numpy scipy`.

After installing the packages, you can use the command below to train the GAT models based on the DGL library

```bash
python dgl_gat_main.py \
    --num_epochs=1000 \
    --hidden_dims=64 \
    --heads=4 \
    --dropout=0.2 \
    --loss_mul=1 \
    --sample_filename=${PATH_TO_INPUT_H5_FILE} \
    --lr=0.001 \
    --seed=1  \
    --cuda \ # if you have GPU available
```



