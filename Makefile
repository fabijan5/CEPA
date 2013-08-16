# Change this to the path to your installed sc-config script.
SCCONFIG = /Users/fabijan/Development/install/mpqc-cpp/bin/sc-config
CXX := $(shell $(SCCONFIG) --cxx)
CXXFLAGS := $(shell $(SCCONFIG) --cxxflags)
CC := $(shell $(SCCONFIG) --cc)
CCFLAGS := $(shell $(SCCONFIG) --cflags)
F77 := $(shell $(SCCONFIG) --f77)
F90 := $(F77)
FFLAGS := $(shell $(SCCONFIG) --f77flags)
F90FLAGS := $(FFLAGS)
CPPFLAGS := $(shell $(SCCONFIG) --cppflags)
LIBS := $(shell $(SCCONFIG) --libs)
LIBDIR  := $(shell $(SCCONFIG) --libdir)
LTCOMP := $(shell $(SCCONFIG) --ltcomp)
LTLINK := $(shell $(SCCONFIG) --ltlink)
LTLINKBINOPTS := $(shell $(SCCONFIG) --ltlinkbinopts)

CPPFLAGS += -I/usr/local/include/Eigen

all : lpno-mp2

lpno-mp2.o: lpno-mp2.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c lpno-mp2.cpp -o lpno-mp2.o
	
lpno-mp2: lpno-mp2.o
	$(LTLINK) $(CXX) $(CXXFLAGS) -o $@ $^ -L$(LIBDIR) $(LIBS) $(LTLINKBINOPTS)

clean:
	-rm -f lpno-mp2 lpno-mp2.o

