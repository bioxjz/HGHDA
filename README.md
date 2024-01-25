<h1 align="center">
  HGHDA
</h1>

Dual-channel hypergraph convolutional network for predicting herb-disease associations
## &#x1F4C2;Experimental Results

1. Contains the results of the experiment and the results of the comparison algorithm such as
   - Randomly_shuffled_graph file is an experimental result for randomly shuffled graphs
   - c_tanimoto_sim file is the model result after calculating the effective component similarity using tanimoto coefficients
   - cold_start file is the result of a cold-start experiment
   - result file is the result of the model and comparison algorithm
## &#x1F4C3;Dataset
Original data can be viewed by clicking on zenodo [here](https://zenodo.org/records/10432947)
## 🚀 Installation

Installation of the project runtime environment，First you can create a virtual environment for the project:
```bash
$ python -m venv [your env name]
```
Activate the virtual environment:
```bash
$ source [your env name]/bin/activate
```
Install project dependencies:
```bash
$ pip install -r requirements.txt
```

## &#x1F3C3; Running
Go to the project directory:
```bash
$ cd Project_Path/src
```
Run the main.py file in the src folder：
```bash
$ python main.py
```
##  🛠️ Configuration
The model can be configured via HGHDA.conf in the src folder
 - datapath:Set the file path of the dataset
 - ratings.setup:Defaults to -columns 0 1 2 (herb,disease,rating)
 - evaluation.setup:Folds for cross validation
 - num.factors:the number of latent factors
 - num.max.epoch:the maximum number of epoch for algorithms.
 - output.setup:the directory path of output results

## &#x1F685; Benchmarks
1. For the baseline code of the comparison algorithm, we give the relevant links.
   - Please refer the code of BiGI [here](https://github.com/caojiangxia/BiGI)
   - Please refer the code of SMGCL [here](https://github.com/Jcmorz/SMGCL)
   - Please refer the code of MilGNet [here](https://github.com/gu-yaowen/MilGNet)
   - Please refer the code of LHGCE [here](https://github.com/shahinghasemi/LHGCE)
   - Please refer the code of HGNNLDA [here](https://github.com/dayunliu/HGNNLDA/tree/main)
2. For the HTInet algorithm, we performed a simple replication of it, using herb-component, component-target, and target-disease association data in experiments. For SMGCL, we use the similarity of effective components to obtain the similarity of herbs and the similarity of targets to obtain the similarity of diseases as their inputs.
3. With respect to the parameter settings involved in running these algorithms, we either explicitly adopted the default settings recommended by their publications or set their associated parameters to the same or similar values as HGHDA.
## ⚖️ License

The code in this package is licensed under the MIT License.
</details>
