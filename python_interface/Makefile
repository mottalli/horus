HEADERS = $(shell find .. -name "*h")
CPPS = $(shell find .. -name "*cpp" | grep -v "main.cpp" | grep -v "prueba_cuda.cpp")
OBJS = $(CPPS:.cpp=.o) horus_cuda_wrap.o ../irisdatabase_kernel.o

CUDASDK = /home/marcelo/Mis_Documentos/Programacion/NVIDIA_GPU_Computing_SDK/C
CXXFLAGS = -O3 -fPIC -I/usr/include/python2.6

_horus_cuda.so:	horus_cuda.py $(OBJS)
	g++ -shared -o _horus_cuda.so $(OBJS) `pkg-config --libs opencv` -L/usr/local/cuda/lib64 -lcudart -L$(CUDASDK)/lib -lcutil_x86_64

horus_cuda_wrap.o:	horus_cuda_wrap.cxx
	g++ $(CXXFLAGS) -c horus_cuda_wrap.cxx 

../irisdatabase_kernel.o:	../irisdatabase_kernel.cu
	nvcc --compiler-options '-fPIC' --shared -O3 -I$(CUDASDK)/common/inc -c  ../irisdatabase_kernel.cu -o ../irisdatabase_kernel.o

horus_cuda_wrap.cxx horus_cuda.py:	$(HEADERS) horus_cuda.i
	swig -c++ -python horus_cuda.i

.cpp.o:
	g++ $(CXXFLAGS) -c $< -o $@
