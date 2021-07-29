# README

The imitation learning network is inherited from with the [CARLA imitation learning Repo](https://github.com/carla-simulator/imitation-learning). Please refer to it for further interest on training details (dataset and setups etc.)

## USAGE
The docker enviroment is highlt recommended for fast and easy setup. 

It is expected that you have some experience with dockers, and have [installed](https://docs.docker.com/install/) and tested your installation to ensure you have GPU access via docker containers.
A quick way to test it is by running:
```bash
# docker >= 19.03
docker run --gpus all,capabilities=utility nvidia/cuda:9.0-base nvidia-smi

# docker < 19.03 (requires nvidia-docker2)
docker run nvidia/cuda:9.0-base --runtime=nvidia nvidia-smi
```
And you should get a standard `nvidia-smi` output.

1. Pull the `AdverseDrive` docker containing all the prerequisite packages for running experiments (also server-friendly)

```bash
docker pull xzgroup/adversedrive:latest
```

2. run the `xzgroup/adversedrive` docker

```bash
docker run -it --rm --gpus 0 --user root --net host -v $(pwd)/:/AdverseDrive -e NB_UID=$(id -u) -e NB_GID=$(id -g) -e GRANT_SUDO=yes xzgroup/adversedrive:latest /bin/bash;
```

3. get output of specfic layers 
```bash
python run_CIL.py
```

4. KL divergence calculation: the softmax layer contains zeros, which may not be considered in `VG_code`. Thus I compute the KL divergence in Python. Run

```bash
python inspect_softmax.py
```
to get KL divergence as well as the `.mat` results.

## DATA PREPARITION 

Original data are from  [Google Drive folder](https://github.com/carla-simulator/imitation-learning). Download the data and put into current path.

For data preprocessing including rotation,  brightness adjustment etc. Please use `mytransfer.py` in VG_code. Change data dirs in `run_CIL.py` to switch between different folder comparision.

## GENERATED SOFTMAX RESULTS

Compressed in `outputs.zip`. `test0, test1, test2.npy` are outputs from origin images (without transformation). Transformations are specified in the name of other arrays.