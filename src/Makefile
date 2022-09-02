default: cuda

cuda:
	g++ -shared -o libhelper_multiprocess.so helper_multiprocess.cpp -I . -fPIC	
	nvcc -o libgpu_kernel.so -shared -Xcompiler -fPIC -lpthread -I .\
	GPUCache.cu GPUMemoryPool.cu Kernels.cu CUDA_IPC_Service.cu \
	GPU_Memory_Node_Storage.cu GPU_Memory_Graph_Storage.cu GPUGraphStore.cu \
	Operator.cu Server.cu \
	-L . -lhelper_multiprocess
main:
	g++ -o gpu-graph-learn main -L . -L /usr/local/cuda/lib64 -lgpu_kernel -lcudart -I .