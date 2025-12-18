CC = gcc
CFLAGS = -Wall -Werror -Wextra -Wpedantic -g -DOP_MODE_BLAS
LBINS = -lm -lopenblas -lcurl -lz

bin/%.o: %.c bin
	$(CC) $(CFLAGS) -c $< -o $@

bin/test%.o: %.c bin
	$(CC) $(CFLAGS) -DTEST -c $< -o $@

all: clean build run
test: clean testbuild run

build: $(addprefix bin/, $(addsuffix .o, $(basename $(wildcard *.c))))
	$(CC) $(CFLAGS) $^ -o bin/out $(LBINS) 

testbuild: $(addprefix bin/test, $(addsuffix .o, $(basename $(wildcard *.c))))
	$(CC) $(CFLAGS) $^ -o bin/out $(LBINS) 

run:
	bin/out

clean:
	rm -rf bin
	rm -rf dataset/*/*

bin:
	mkdir bin
