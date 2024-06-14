#!/bin/bash
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

# go to https://github.com/triton-inference-server/tensorrtllm_backend and check available branches/tags
VERSION_TAG=$1
TARGET_DIR=$2

echo "----------------------------------"
echo " VERSION_TAG ${VERSION_TAG} "
echo " TARGET_DIR ${TARGET_DIR} "
echo "----------------------------------"

if [ -d "$TARGET_DIR" ]; then
    cd ${TARGET_DIR}
else
    echo "Directory does not exist. Use home dir"
    cd ${HOME}
fi

echo "----------------------------------"
echo "git clone source code"
git clone https://github.com/triton-inference-server/tensorrtllm_backend
cd ./tensorrtllm_backend
git checkout ${VERSION_TAG}
git lfs install
git submodule update --init --recursive
echo "----------------------------------"
echo "Build docker"
# For x86_64
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm:${VERSION_TAG} -f dockerfile/Dockerfile.trt_llm_backend .
