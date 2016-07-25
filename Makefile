# - copy&paste this to any new Makefile, !!! modify include !!!
# autodetect <srcdir> as the path to this Makefile
ifndef srcdir
 srcdir:=$(dir $(lastword $(MAKEFILE_LIST)))
endif
# include the platform header
# - it includes config.make and may overwrite anything set before
include $(srcdir)make/build.make
# - now we can use "include $(makedir)..."
# - platform initialized, end of copy&paste 

# -- Flann wrapper --

FLANN:=$(call em_link_bin,flann,$(call em_compile,$(srcdir)src/flann.cpp))
FLANN+=$(call em_link_bin,flann-train,$(call em_compile,$(srcdir)src/flann-train.cpp))
FLANN+=$(call em_link_bin,flann-predict,$(call em_compile,$(srcdir)src/flann-predict.cpp))

$(FLANN):PACKAGES:=argtable2 opencv
$(FLANN):FLAGS:=-std=c++14

all:$(FLANN)

$(call em_install,flann,$(FLANN))

# -- Data normalization tool

NORMALIZE:=$(call em_link_bin,normalize,$(call em_compile,$(srcdir)src/normalize.cpp))

$(NORMALIZE):FLAGS:=-std=c++14

all:$(NORMALIZE)

$(call em_install,normalize,$(NORMALIZE))
