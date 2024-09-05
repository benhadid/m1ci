#CXX=clang++-12
#CXX=icpx
CXX=g++
CXXFLAGS=-march=native -mtune=native -mpclmul -O3 -std=c++11 -Wall -fPIC -fvisibility=hidden -fno-strict-aliasing 

INCLUDES=-I./include 
DEFINES=-DFPNG_NO_SSE=0 -D_LARGEFILE64_SOURCE=1 -D_FILE_OFFSET_BITS=64 

APP_NAME=mnt

OBJDIR=objs
SRCDIR=src

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
	/bin/mkdir -p $(OBJDIR)/

clean:
	/bin/rm -rf $(OBJDIR) *~ *.png $(APP_NAME) 

OBJS=$(OBJDIR)/mnt.o $(OBJDIR)/fpng.o  $(OBJDIR)/main.o

$(APP_NAME): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) $(DEFINES) $(INCLUDES) -c -o $@

