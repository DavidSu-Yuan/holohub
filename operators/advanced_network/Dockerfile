FROM nvcr.io/nvidia/clara-holoscan/holoscan:v0.5.0 

ARG HOLOHUB_DIR=/opt/holohub-internal
ARG OFED_VERSION=5.8-1.0.1.1
ARG UBUNTU_VERSION=20.04
ARG ADV_NET_DIR=holohub-internal/operators/advanced_network/
RUN wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-$OFED_VERSION/MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-aarch64.tgz && \
        tar xvf MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-aarch64.tgz && \
        cd MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-aarch64 && \
        ./mlnxofedinstall --upstream-libs --dpdk --with-mft --upstream-libs --user-space-only --force --without-fw-update && \
        cd ../ && \
        rm -fr MLNX_OFED_LINUX-$OFED_VERSION-ubuntu$UBUNTU_VERSION-aarch64 && \
        rm -rf /var/lib/apt/lists/*

ARG DPDK_VERSION=22.11.1
RUN apt update && apt install -y python3-pyelftools ninja-build meson 

ADD https://fast.dpdk.org/rel/dpdk-${DPDK_VERSION}.tar.xz /tmp/
RUN cd /tmp && tar xf dpdk-${DPDK_VERSION}.tar.xz
COPY ${ADV_NET_DIR}/dpdk_patches/*.patch /tmp/dpdk-stable-${DPDK_VERSION}
WORKDIR /tmp/dpdk-stable-${DPDK_VERSION}/
RUN patch --ignore-whitespace --fuzz 3  config/arm/meson.build /tmp/dpdk-stable-${DPDK_VERSION}/dpdk.nvidia.patch 
RUN patch --ignore-whitespace --fuzz 3  drivers/gpu/cuda/devices.h /tmp/dpdk-stable-${DPDK_VERSION}/devices.h.patch
RUN patch --ignore-whitespace --fuzz 3  drivers/gpu/cuda/cuda.c /tmp/dpdk-stable-${DPDK_VERSION}/cuda.c.patch
RUN CFLAGS=-I/usr/local/cuda/include meson build -Dplatform=generic -Dc_args=-I/usr/local/cuda/include \
          -Ddisabled_drivers=baseband/*,bus/ifpga/*,common/cpt,common/dpaax,common/iavf,common/octeontx,common/octeontx2,crypto/nitrox,net/ark,net/atlantic,net/avp,net/axgbe,net/bnx2x,net/bnxt,net/cxgbe,net/e1000,net/ena,net/enic,net/fm10k,net/hinic,net/hns3,net/i40e,net/ixgbe,vdpa/ifc,net/igc,net/liquidio,net/netvsc,net/nfp,net/qede,net/sfc,net/thunderx,net/vdev_netvsc,net/vmxnet3,regex/octeontx2,
RUN ninja -C build install
RUN /tmp/dpdk-${DPDK_VERSION}.tar.xz

ADD holohub-internal ${HOLOHUB_DIR}
WORKDIR ${HOLOHUB_DIR}
RUN ls -lart ${HOLOHUB_DIR}
RUN ./run build advanced_network

WORKDIR /tmp/
ARG NATS_VER=v2.9.8
RUN wget https://github.com/nats-io/nats-server/releases/download/${NATS_VER}/nats-server-${NATS_VER}-arm64.deb && \
        dpkg -i nats-server-${NATS_VER}-arm64.deb
RUN rm /tmp/nats-server-${NATS_VER}-arm64.deb

RUN pip3 install scipy plotly plotly_express matplotlib dash dash_bootstrap_components nats-py loguru attrs

