TARGET = prog.out
PROGRAM_NAME = prog.out
LIBS = -lm
CC = gcc
CFLAGS = -g -Wall -fopenmp

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.c, %.o, $(wildcard *.c))
HEADERS = $(wildcard *.h)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -Wall -fopenmp $(LIBS) -o $@

run:
	./${PROGRAM_NAME} ${ARGS}

clean:
	-rm -f *.o
	-rm -f $(TARGET)
