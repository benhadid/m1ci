CXX=g++ -march=native -mtune=native
CXXFLAGS=-I./include -I./objs -O0 -std=c++11 -Wall -fPIC -fopenmp
ISPC=ispc
# note: requires AVX2
# disabling AVX2 FMA since it causes a difference in output compared to reference on Mandelbrot 
#ISPCFLAGS=-O3 --target=avx2-i32x8 --arch=x86-64 --opt=disable-fma --pic
ISPCFLAGS=-O3 --target=sse4.1-i32x8 --arch=x86-64 --pic

APP_NAME=kmeans

OBJDIR=objs
SRCDIR=src

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *~ $(APP_NAME)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/naive_version.o $(OBJDIR)/refactored_version.o $(OBJDIR)/openmp_version.o $(OBJDIR)/utils.o $(OBJDIR)/argmin_ispc.o 

$(APP_NAME): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/openmp_version.o: $(OBJDIR)/argmin_ispc.h

$(OBJDIR)/%_ispc.h $(OBJDIR)//%_ispc.o: $(SRCDIR)/%.ispc
	$(ISPC) $(ISPCFLAGS) $< -o $(OBJDIR)/$*_ispc.o -h $(OBJDIR)/$*_ispc.h	