# Mirror image: registry.cn-shanghai.aliyuncs.com/pai-kubeflow/nogfw:tensorflow-1.13.0rc2-gpu
FROM tensorflow/tensorflow:1.13.0rc2-gpu
USER root

#install wget
RUN apt update && apt install -y wget vim git

#openssl required by grpc
RUN cd /root \
&& wget https://www.openssl.org/source/old/1.0.2/openssl-1.0.2k.tar.gz \
&& tar -xzvf openssl-1.0.2k.tar.gz \
&& cd openssl-1.0.2k \
&& ./config shared zlib-dynamic \
&& make -j 32 \
&& mkdir -p /usr/lib64 \
&& cp libcrypto.so.1.0.0 /usr/lib64/ \
&& cp libssl.so.1.0.0 /usr/lib64/ \
&& cd /usr/lib64 \
&&  ln -s libssl.so.1.0.0 libssl.so.10 \
&&  ln -s libcrypto.so.1.0.0 libcrypto.so.10
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

#install graph-learn
RUN pip install http://graph-learn-whl.oss-cn-zhangjiakou.aliyuncs.com/graphlearn-2.1dev_1529ad0-cp27-cp27mu-linux_x86_64.whl

# download code
RUN mkdir -p /workspace && cd /workspace && git clone https://github.com/alibaba/graph-learn.git
