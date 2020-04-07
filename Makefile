SRCS := main.cpp

OPENMP := -fopenmp
ARGS := -std=c++14 $(OPENMP)
all: 
	g++ $(ARGS) $(SRCS) -o aqc
