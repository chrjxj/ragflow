#!/usr/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


HF_MODEL_DIR=$1
TRT_ENGINE_DIR=$2
MAX_BATCH_SIZE=$3
TP_SIZE=$4


echo "----------------------------------"
echo " Command line params "
echo " HF_MODEL_DIR ${HF_MODEL_DIR} "
echo " TRT_ENGINE_DIR ${TRT_ENGINE_DIR} "
echo " MAX_BATCH_SIZE ${MAX_BATCH_SIZE} "
echo " TP_SIZE ${TP_SIZE} "
echo "----------------------------------"

echo "----------------------------------"
echo " convert model "
echo "----------------------------------"
if ! ls ${TRT_ENGINE_DIR}/*.engine >/dev/null 2>&1; then

    # updated for trtllm v0.9 or above
	echo "Start model converstion"
    mkdir -p ${TRT_ENGINE_DIR}
    mkdir -p /tmp/trt_ckpts
    # Check if input variable is greater than 1

    python /opt/tensorrt_llm/examples/llama/convert_checkpoint.py \
                --model_dir $HF_MODEL_DIR \
                --dtype float16 \
                --output_dir /tmp/trt_ckpts \
                --tp_size ${TP_SIZE}

    trtllm-build --checkpoint_dir /tmp/trt_ckpts \
                --output_dir ${TRT_ENGINE_DIR} \
                --gemm_plugin float16 \
                --max_batch_size=$MAX_BATCH_SIZE \
                --max_input_len=3072 \
                --max_output_len=1024
    rm -rf /tmp/trt_ckpts
else
	echo "Skip model converstion. Found engine file in ${TRT_ENGINE_DIR}"
fi


echo "----------------------------------"
echo " config ensemble_models repo for triton server "
echo "----------------------------------"

fill_triton_repo_streaming () {
    mkdir -p /ensemble_models && rm -rf /ensemble_models/llama_ifb
    cp -rf /opt/ensemble_models/llama_ifb  /ensemble_models/llama_ifb        
    # Modify config.pbtxt
    python3 /opt/scripts/fill_template.py -i /ensemble_models/llama_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_DIR},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:1
    python3 /opt/scripts/fill_template.py -i /ensemble_models/llama_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_DIR},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:1
    python3 /opt/scripts/fill_template.py -i /ensemble_models/llama_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:True,bls_instance_count:1,accumulate_tokens:False
    python3 /opt/scripts/fill_template.py -i /ensemble_models/llama_ifb/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE}
    python3 /opt/scripts/fill_template.py -i /ensemble_models/llama_ifb/tensorrt_llm/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:True,max_beam_width:1,engine_dir:${TRT_ENGINE_DIR},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:600
}

fill_triton_repo () {
    # wget https://raw.githubusercontent.com/triton-inference-server/tensorrtllm_backend/main/tools/fill_template.py -O /opt/scripts/fill_template.py
    mkdir -p /ensemble_models && rm -rf /ensemble_models/llama_ifb
    cp -rf /opt/ensemble_models/llama_ifb  /ensemble_models/llama_ifb        

    python3 /opt/scripts/fill_template.py -i /ensemble_models/llama_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_DIR},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:1
    python3 /opt/scripts/fill_template.py -i /ensemble_models/llama_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL_DIR},tokenizer_type:auto,triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:1
    python3 /opt/scripts/fill_template.py -i /ensemble_models/llama_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
    python3 /opt/scripts/fill_template.py -i /ensemble_models/llama_ifb/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE}
    python3 /opt/scripts/fill_template.py -i /ensemble_models/llama_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:False,max_beam_width:1,engine_dir:${TRT_ENGINE_DIR},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0
}


fill_triton_repo_streaming
