CUDA_DIR := /opt/cuda
MATLAB_DIR := /usr/local/MATLAB/R2013b
PROJECT_NAME :=ComputeFeature
EXT := $(shell $(MATLAB_DIR)/bin/mexext) 
NVCC := $(CUDA_DIR)/bin/nvcc 
MEX := $(MATLAB_DIR)/bin/mex
CXX := /usr/bin/g++
# Common includes and paths for CUDA
INCLUDES_DIR  := $(CUDA_DIR)/include
COMMONFLAGS := -I$(INCLUDES_DIR)
##COMMONFLAGS += $(foreach indir,$(INCLUDES_DIR),-I$(indir))
LIBARIES := cudart #cublas curand

CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifneq ("$(wildcard $(CUDA_DIR)/lib64)","")
	CUDA_LIB_DIR += $(CUDA_DIR)/lib64
endif
LDFLAGS += $(foreach libdir,$(CUDA_LIB_DIR),-L$(libdir)) \
	   $(foreach lib,$(LIBARIES),-l$(lib))

CXXFLAGS := $(COMMONFLAGS)
#CXXFLAGS := -fPIC $(COMMONFLAGS)
MEXFLAGS := $(CXXFLAGS) -Wno-uninitialized
NVCCFLAGS := -ccbin=$(CXX) -Xcompiler -fPIC $(COMMONFLAGS)
## static link
#STATIC := compute_hist.a
#STATIC_LINK_COMMAND := -Wl,--whole-archive $(STATIC) -Wl,--no-whole-archive
STATIC_LINK_COMMAND := -Wl,--no-whole-archive
################################################################################

# CUDA code generation flags
#GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM32    := -gencode arch=compute_32,code=sm_32
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SMXX    := -gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?= $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX)

################################################################################

# Target rules
all: mex$(PROJECT_NAME).$(EXT)

#compute_hist.a: compute_hist.o
#	ar rcs $@ compute_hist.o

$(PROJECT_NAME).o: $(PROJECT_NAME).cu
	@ echo $(NVCC) $<
	@ $(NVCC)  $(NVCCFLAGS)  $(GENCODE_FLAGS) -o $@ -c $<

mex$(PROJECT_NAME).$(EXT): mex$(PROJECT_NAME).cpp $(PROJECT_NAME).o
	@ echo $(MEX) $<
	@ $(MEX) $^\
	        CXX="$(CXX)" \
		CXXFLAGS="\$$CXXFLAGS $(MEXFLAGS)" \
		CXXLIBS="\$$CXXLIBS $(STATIC_LINK_COMMAND) $(LDFLAGS)" -output $@ 
	rm -f $(PROJECT_NAME).o
print:
	@ echo $(MEXFLAGS)
	@ echo $(CUDA_LIB_DIR)
#run: build
#	./compute_hist

clean:
	rm -f compute_hist.a $(PROJECT_NAME).o mex$(PROJECT_NAME).$(EXT)

