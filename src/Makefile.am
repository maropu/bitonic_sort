# install bsort to bin directory
bin_PROGRAMS = bsort

# list of sources of bsort
bsort_SOURCES = main.c bitonic_sort.c err_utils.c

# specify flags
CFLAGS = -Wall -O3 -funroll-loops -msse2 -mtune=nocona

# specify libraries
LIBS = -lm -lpthread

# specify include directory
INCLUDES = -I$(top_srcdir)/include
