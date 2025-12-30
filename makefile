CC = gcc
CFLAGS = -Wall -Werror -Wextra -Wpedantic -g -DOP_MODE_BLAS
LBINS = -lm -lopenblas -lcurl -lz
SRCS = $(filter-out test.c mnistdemo.c, $(wildcard *.c))

bin/%.o: %.c bin
	$(CC) $(CFLAGS) -c $< -o $@

bin/test%.o: %.c bin
	$(CC) $(CFLAGS) -DTEST -c $< -o $@

testtarget: clean bin buildtest run
mnisttarget: clean bin buildmnistdemo run

buildtest: bin/test.o $(addprefix bin/test, $(addsuffix .o, $(basename $(SRCS))))
	$(CC) $(CFLAGS) $^ -o bin/out $(LBINS) 

buildmnistdemo: bin/mnistdemo.o $(addprefix bin/, $(addsuffix .o, $(basename $(SRCS))))
	$(CC) $(CFLAGS) $^ -o bin/out $(LBINS) 

run:
	bin/out

clean:
	rm -rf bin

bin:
	mkdir bin
