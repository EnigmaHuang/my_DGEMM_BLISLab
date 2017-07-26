EXENAME   = my_dgemm

CC        = g++
GCCFLAGS  = -O1 -pg -g -fopenmp -m64 -march=native
ICCFLAGS  = -O3 -qopenmp -m64 -xHost  
INCFLAGS  = -I /home/enigma/OpenBLAS-0.2.19-gcc/include/ 
LDFLAGS   = -L/home/enigma/OpenBLAS-0.2.19-gcc/lib -lopenblas
CFLAGS    = $(INCFLAGS) $(GCCFLAGS)

CPPSRCS   = $(wildcard *.cc)
OBJS      = $(CPPSRCS:.cc=.o)

build : $(EXENAME)

$(EXENAME): $(OBJS) 
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(EXENAME) $(OBJS)

%.o: %.cc
	$(CC) $(CFLAGS) -c $^

clean:
	$(RM) *.o $(EXENAME)
