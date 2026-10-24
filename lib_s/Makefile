CC = mpicc
CFLAGS = -O2 -Wall
# Добавляем путь к вашей установке ScaLAPACK
LDFLAGS = -L$(HOME)/scalapack-install/lib
# Убираем -lblacs (она падает) и добавляем -lgfortran, как в вашей рабочей команде
LIBS = -lscalapack -llapack -lblas -lgfortran -lm

all: task1

simulation.o: simulation.c simulation.h
	$(CC) $(CFLAGS) -c simulation.c -o simulation.o

main.o: main.c simulation.h
	$(CC) $(CFLAGS) -c main.c -o main.o

# Добавляем $(LDFLAGS) перед $(LIBS) при линковке
task1: main.o simulation.o
	$(CC) $(CFLAGS) main.o simulation.o -o task1 $(LDFLAGS) $(LIBS)

clean:
	rm -f *.o task1
