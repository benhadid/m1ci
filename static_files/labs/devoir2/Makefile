#CXX=g++
CXX=icpx
CXXFLAGS=-march=native -mtune=native -I./include -I./objs -O0 -std=c++11 -Wall -fPIC -fiopenmp
ISPC=ispc
# note: requires AVX2
# disabling AVX2 FMA since it causes a difference in output compared to reference on Mandelbrot 
#ISPCFLAGS=-O3 --target=avx2-i32x8 --arch=x86-64 --opt=disable-fma --pic
ISPCFLAGS=-O3 --target=sse4.1-i32x8 --arch=x86-64 --pic 

APP_NAME=matmult

OBJDIR=objs
SRCDIR=src

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *~ $(APP_NAME)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/matrixMultiply_ispc.o

$(APP_NAME): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: $(OBJDIR)/matrixMultiply_ispc.h

$(OBJDIR)/%_ispc.h $(OBJDIR)//%_ispc.o: $(SRCDIR)/%.ispc
	$(ISPC) $(ISPCFLAGS) $< -o $(OBJDIR)/$*_ispc.o -h $(OBJDIR)/$*_ispc.h	
