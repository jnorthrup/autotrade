# Makefile for ANE HRM Training

CC = xcrun clang
CFLAGS = -O2 -framework Foundation -framework IOSurface \
         -framework CoreML -framework Accelerate -ldl -lobjc \
         -I./ANE/training

TARGET = ane_hrm_train
SOURCE = ane_hrm_train.m

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)
	@echo "Built $(TARGET)"

clean:
	rm -f $(TARGET)

.PHONY: all clean
