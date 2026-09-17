FROM ubuntu:22.04

# install system dependencies
# (ldap-utils, libnss-ldapd, libpam-ldapd, nscd, nslcd are needed to resolve
#  training users against the environment's LDAP, as in the other apps)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        curl \
        git \
        dnsutils \
        htop \
        locales \
        ldap-utils \
        libnss-ldapd \
        libpam-ldapd \
        less \
        nano \
        nodejs \
        nscd \
        nslcd \
        pciutils \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        rsync \
        time \
        unzip \
        vim \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_NZ -c -f UTF-8 -A /usr/share/locale/locale.alias en_NZ.UTF-8

ENV LANG en_NZ.utf8

# install kubectl, required for running on the k8s cluster
ARG KUBECTL_VERSION=v1.28.5
RUN curl -LO https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl \
    && mv kubectl /usr/local/bin/kubectl \
    && chmod +x /usr/local/bin/kubectl

# copy in ood k8s utils
ARG UTILS_HASH=6298fb01f7a7c66a8454e3f0fd74437a32491423
RUN git clone https://github.com/nesi/training-environment-k8s-utils.git /opt/ood-k8s-utils \
    && cd /opt/ood-k8s-utils \
    && git checkout $UTILS_HASH \
    && chmod +x /opt/ood-k8s-utils/files/* \
    && mv /opt/ood-k8s-utils/files/* /bin/ \
    && rm -rf /opt/ood-k8s-utils

# ---------------------------------------------------------------------------
# nvtop
#
# Built from source rather than installed from apt. The jammy package is 1.x,
# which predates the dlopen-based NVIDIA backend this image relies on, and it
# also drags in a large set of AMD/Intel GPU dependencies we have no use for.
# Building with only the NVIDIA backend enabled keeps it small and guarantees
# it goes through libnvidia-ml.so.1, which is what our shim provides.
# ---------------------------------------------------------------------------
ARG NVTOP_VERSION=3.1.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        cmake libncurses5-dev libncursesw5-dev libdrm-dev \
    && git clone --depth 1 --branch $NVTOP_VERSION https://github.com/Syllo/nvtop.git /tmp/nvtop \
    && mkdir -p /tmp/nvtop/build \
    && cd /tmp/nvtop/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DNVIDIA_SUPPORT=ON \
        -DAMDGPU_SUPPORT=OFF \
        -DINTEL_SUPPORT=OFF \
        -DMSM_SUPPORT=OFF \
        -DPANFROST_SUPPORT=OFF \
        -DPANTHOR_SUPPORT=OFF \
        -DASCEND_SUPPORT=OFF \
        -DTPU_SUPPORT=OFF \
    && make -j"$(nproc)" \
    && make install \
    && cd / && rm -rf /tmp/nvtop \
    && apt-get purge -y cmake && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python stack for the workshop
#
# This comes before the emulator source is copied so that editing the emulator
# does not invalidate a two-gigabyte download. torch is the CPU build: the CUDA
# build would be far larger and could not run anyway, since there is no device.
# gpuemu.torch_shim is what makes CUDA-flavoured code work on top of it.
# ---------------------------------------------------------------------------

# Ubuntu 22.04 ships setuptools 59, which predates PEP 621 and would silently
# install gpuemu as a package called "UNKNOWN" instead of reading [project]
# from pyproject.toml. Upgrading first is what makes that install work.
RUN pip3 --no-cache-dir install --upgrade pip 'setuptools>=64' wheel

RUN pip3 --no-cache-dir install \
        jupyterlab \
        jupyterlab-nvdashboard \
        notebook \
        ipywidgets \
        matplotlib \
        numpy \
        pandas \
        nvidia-ml-py \
    && pip3 --no-cache-dir install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip3 --no-cache-dir install numba pytest

# Numba's CUDA simulator runs @cuda.jit kernels in the interpreter, so learners
# can write and debug real CUDA kernel code - threads, blocks, shared memory,
# syncthreads - and have it execute correctly, if slowly.
ENV NUMBA_ENABLE_CUDASIM=1

# ---------------------------------------------------------------------------
# The GPU emulator
#
# libnvidia-ml.so.1 is installed to /opt/gpuemu/lib and put on the loader path
# through /etc/ld.so.conf.d. Everything that looks for a GPU - nvtop, pynvml,
# anything else that dlopens NVML - finds it there and sees an L4.
# ---------------------------------------------------------------------------
COPY gpuemu /opt/gpuemu/src

RUN cd /opt/gpuemu/src/nvml \
    && make PREFIX=/opt/gpuemu install \
    && echo "/opt/gpuemu/lib" > /etc/ld.so.conf.d/gpuemu.conf \
    && ldconfig \
    && cd /opt/gpuemu/src \
    && pip3 --no-cache-dir install . \
    && install -m 0755 /opt/gpuemu/src/bin/nvidia-smi /usr/local/bin/nvidia-smi \
    && install -d /run/gpuemu /run/gpuemu/claims /run/gpuemu/slurm/jobs \
    && chmod 1777 /run/gpuemu /run/gpuemu/claims /run/gpuemu/slurm /run/gpuemu/slurm/jobs

# Verify the emulator at build time rather than discovering it in front of a
# workshop. The C and Python views of the shared state must agree, the shim must
# load and enumerate a device, torch must see the emulated card and refuse an
# oversized allocation, and a real @cuda.jit kernel must execute.
RUN cd /opt/gpuemu/src \
    && python3 -m pytest tests -q \
    && gpuemud --once \
    && nvidia-smi -L \
    && nvidia-smi \
    && python3 -c "import ctypes; l=ctypes.CDLL('libnvidia-ml.so.1'); assert l.nvmlInit_v2()==0; n=ctypes.c_uint(); assert l.nvmlDeviceGetCount_v2(ctypes.byref(n))==0; assert n.value==1, n.value; print('NVML shim OK')" \
    && python3 -c "\
import numpy as np; from numba import cuda;\
exec('@cuda.jit\ndef add(a,b,o):\n i=cuda.grid(1)\n if i<o.size: o[i]=a[i]+b[i]\n');\
a=np.arange(64,dtype=np.float32); b=a*2; o=np.zeros_like(a);\
add[2,32](a,b,o);\
assert np.allclose(o,a+b), o;\
print('numba cudasim OK')" \
    && pip3 uninstall -y pytest \
    && rm -rf /run/gpuemu/state.bin /opt/gpuemu/src/.pytest_cache

# Claim the device automatically when a process imports torch or numba.cuda, so
# a learner's notebook shows up in nvidia-smi without any boilerplate.
ENV GPUEMU_AUTOCLAIM=1
ENV GPUEMU_STATE_FILE=/run/gpuemu/state.bin
ENV GPUEMU_CLAIMS_DIR=/run/gpuemu/claims
ENV GPUEMU_SLURM_DIR=/run/gpuemu/slurm

# sitecustomize runs in every Python process, which is what makes the two
# settings above take effect in notebook kernels and batch jobs alike.
RUN printf '%s\n' \
    'try:' \
    '    from gpuemu.autoclaim import install as _gpuemu_autoclaim' \
    '    _gpuemu_autoclaim()' \
    'except Exception:' \
    '    pass' \
    > "$(python3 -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py"

# ---------------------------------------------------------------------------
# Workshop material
# ---------------------------------------------------------------------------
COPY notebooks /opt/gpu-training/notebooks
COPY examples /opt/gpu-training/examples
RUN chmod -R o+rX /opt/gpu-training

# A short banner so anyone opening a terminal knows the GPU is emulated. Being
# quiet about that would be the one genuinely harmful thing this image could do.
COPY profile.d/gpuemu.sh /etc/profile.d/gpuemu.sh
RUN chmod 0644 /etc/profile.d/gpuemu.sh
