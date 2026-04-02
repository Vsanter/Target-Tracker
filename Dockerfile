# TrackFormer Dockerfile (完整版 - CPU 可编译)
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 配置系统源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装基础依赖
RUN apt-get update || apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        curl \
        git \
        vim \
        openssh-server \
        sudo \
        ca-certificates \
        cmake \
        && rm -rf /var/lib/apt/lists/*

# 配置 pip 源
RUN mkdir -p /root/.pip && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url = https://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf && \
    echo "trusted-host = mirrors.aliyun.com" >> /root/.pip/pip.conf && \
    echo "timeout = 300" >> /root/.pip/pip.conf

WORKDIR /app

# CUDA 环境变量（保留 conda 路径）
ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib:/opt/conda/lib/python3.7/site-packages/torch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# 配置 SSH
RUN mkdir -p /var/run/sshd && \
    echo 'root:password' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

ENV VSCODE_SKIP_REQUIREMENTS_CHECK=1

# 复制项目（先复制 setup.py 和需要编译的文件）
COPY src/trackformer/models/ops/setup.py src/trackformer/models/ops/setup.py
COPY src/trackformer/models/ops/src/ src/trackformer/models/ops/src/

# 编译安装 MultiScaleDeformableAttention (CPU 版本)
RUN python src/trackformer/models/ops/setup.py build install

# 复制剩余项目文件
COPY . .

# 安装 pycocotools
RUN pip install --no-cache-dir -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'

# 安装 motmetrics 和 lapsolver
RUN pip install --no-cache-dir motmetrics lapsolver

# 安装 ninja
RUN pip install --no-cache-dir ninja

# 安装项目
RUN pip install --no-cache-dir -e .

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
