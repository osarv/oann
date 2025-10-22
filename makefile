CC = gcc
CFLAGS = -Wall -Werror -Wextra -Wpedantic -g
LBINS = -lm

bin/%.o: %.c bin
	$(CC) $(CFLAGS) -c $< -o $@

all: clean build run

build: $(addprefix bin/, $(addsuffix .o, $(basename $(wildcard *.c))))
	$(CC) $(CFLAGS) $^ -o bin/out $(LBINS) 

run:
	bin/out

clean:
	rm -rf bin

bin:
	mkdir bin
