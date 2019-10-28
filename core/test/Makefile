ROOT_DIR=..
INCLUDE_DIR=${ROOT_DIR}/include
SRC_DIR=${ROOT_DIR}/src
INCLUDES=-I${INCLUDE_DIR}
FLAGS=-O3 ${INCLUDES}

all: wisard_test
wisard_test: wisard_test.o bitarray.o wisard.o
	g++ -std=c++11 ${FLAGS} wisard_test.o bitarray.o wisard.o -o wisard_test -Wall
wisard_test.o: wisard_test.c
	gcc ${FLAGS} -c wisard_test.c -o wisard_test.o -Wall
wisard.o: ${SRC_DIR}/wisard.cc ${INCLUDE_DIR}/wisard.h
	g++ -std=c++11 ${FLAGS} -c ${SRC_DIR}/wisard.cc -o wisard.o -Wall
bitarray.o: ${SRC_DIR}/bitarray.cc ${INCLUDE_DIR}/bitarray.h
	gcc ${FLAGS} -c ${SRC_DIR}/bitarray.cc -o bitarray.o -Wall

clean:
	rm *.o wisard_test