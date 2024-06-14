# Deploy Local LLM Inference using NVIDIA TenosrRT-LLM


## Setup

Before start, you need to prepare and setup followings:

- Have a GPU server, with GPU Driver 550.54.15 and CUDA 12.4 installed
- On your GPU server, install Docker-CE. See: https://docs.docker.com/engine/install/
- On your GPU server, Install NVIDIA Container Toolkit，See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Go to https://catalog.ngc.nvidia.com, register an account for free, and generate an API key (upright, choose "setup")
- On your GPU server, login to `nvcr.io` using your NGC token:

```shell
$ docker login nvcr.io
Username: $oauthtoken
Password: <Your API Key>
```

- After login `nvcr.io`, you should be able to pull docker images, e.g. `docker pull nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3`
- Download huggingface models to your GPU server, e.g. `/models` folder on your host machine

```shell
$ ls /models
 Llama2-13b-chat-hf  Qwen2-7B
```

#### Build Docker image

build a docker image and tag it as `tritonserver-trtllm:24.05-0.9.0`

```bash
cd ragflow
docker build -f Dockerfile.trtllm -t tritonserver-trtllm:24.05-0.9.0 .
```

Note: 
- See https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver for different tags
- Please check related TensorRT-LLM/TritonServer versions in the release notes: https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html

## LLM Inference Examples

This section shows how to deploy LLM models locally using NVIDIA TensorRT-LLM and Triton Inference Server, on NVIDIA GPUs.

The following is tested with GPU Driver Version 550.54.15 and CUDA Version 12.4

### Llama2

**1 Deploy** 

On your host, assume you have downloaded llama model to `/models/Llama2-13b-chat-hf` folder.

On your host, start docker: 

```bash
docker run -it --rm --runtime=nvidia --gpus  '"device=0"'  \
    -v /models:/models \
    --network=host --ipc=host  \
    --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    tritonserver-trtllm:24.05-0.9.0 \
    bash
```

Inside the docker container, you can build tensorrt engines and config ensemble models, using following script:

```bash
# note - "/models/Llama2-13b-chat-hf" should contain huggingface models
# note - "/models/Llama2-13b-chat-hf/trt_engines-fp16tp4" folder of tensorrt engines
# note - batch size is 8, and tp size (tensor parallel size) is 1
/opt/model_server/deploy_llama.sh  /models/Llama2-13b-chat-hf /models/Llama2-13b-chat-hf/trt_engines-fp16tp1  8 1
```


**2 Start Server** 

Inside the docker container, Use either way to start triton server (grpc and streaming mode): 

```bash
/opt/tritonserver/bin/tritonserver --model-repository=/ensemble_models/llama_ifb  --allow-http False --allow-grpc True --log-verbose=2
```

or:
```bash
python3 /opt/scripts/launch_triton_server.py  --model_repo=/ensemble_models/llama_ifb --world_size 1 --grpc_port 8001 --http_port 8000
```

**3 Sanity Check** 

Inside the same docker container (or any Python enviroment with `tritionclient[all]` package installed), you can send client requrest to triton server: 

```bash
python3 /opt/model_client/trt_llm.py --port 8001 --question "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. what is the fastest fish in river?"
```

### Qwen2

The workflow and steps are similar to above Llama2 section.

You will need to use TensorRT-LLM v0.10.0 or above to deploy Qwen2 model.

**0 Build Docker image** 

At 2024/06/10, we need to manually build docker image for TensorRT-LLM v0.10.0.

Build 1st docker image and tag it as `triton_trt_llm:v0.10.0`

```bash
cd ragflow/docker/trtllm
chmod +x build_docker.sh
./build_docker.sh v0.10.0 $HOME
```

Based on `triton_trt_llm:v0.10.0` and build 2nd docker image and tag it as `tritonserver-trtllm:v0.10.0`:

```
cd ragflow
docker build -f Dockerfile.trtllm_v0.10.0 -t tritonserver-trtllm:v0.10.0 .
```

On your host, start docker: 

```bash
docker run -it --rm --runtime=nvidia --gpus  '"device=0,1,2,3"'  \
    -v /models:/models \
    --network=host --ipc=host  \
    --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
    tritonserver-trtllm:v0.10.0 \
    bash

```

#### Use 1 GPU for TP_Size=1

**1 Deploy** 

On your host, assume you have downloaded huggingface model to `/models/Qwen2-7B` folder.

Inside the docker container, you can build tensorrt engines and config ensemble models, using following script. 

```bash
# batch size is 8, and tp size (tensor parallel size) is 1
./deploy_qwen.sh /models/Qwen2-7B /models/Qwen2-7B/trt_engines-fp16tp1   8 1
```

**2 Start Server** 

Inside the docker container, Use either way to start triton server (grpc and streaming mode): 

```bash
/opt/tritonserver/bin/tritonserver --model-repository=/ensemble_models/qwen_ifb  --allow-http False --allow-grpc True --log-verbose=2
```
or:
```bash
python3 /opt/scripts/launch_triton_server.py  --model_repo=/ensemble_models/qwen_ifb --world_size 1 --grpc_port 8001
```

**3 Sanity Check** 

Inside the same docker container (or any Python enviroment with `tritionclient[all]` package installed), you can send client requrest to triton server: 

```bash
python3 /opt/model_client/trt_llm.py --port 8001 --question "你是一个知识助手，请简要回答问题。陆地上最快的动物是什么?"
```


#### Use 4 GPUs for TP_Size=4

If you want to use multiple GPUs to deploy models like Qwen2-7B/14B/72B, please use tensor parallism for inference. Deployment steps are similar to tp_size = 1. See details below. 

**1 Deploy** 

On your host, assume you have downloaded huggingface model to `/models/Qwen2-7B` folder.

Inside the docker container, you can build tensorrt engines and config ensemble models, using following script. 
```bash
# batch size is 8, and tp size (tensor parallel size) is 4
# you need 4 GPUs 
./deploy_qwen.sh /models/Qwen2-7B /models/Qwen2-7B/trt_engines-fp16tp4   8 4
```

**2 Start Server** 

Inside the docker container, start triton server: 
```bash
python3 /opt/scripts/launch_triton_server.py  --model_repo=/ensemble_models/qwen_ifb --world_size 4 --grpc_port 8001
```

Above script will use 'mpirun' to start multiple triton server processes. If you want to stop the server, you can find the 'mpirun' process and kill it.

