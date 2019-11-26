export PATH := /usr/local/cuda-9.0/bin:$(PATH)

all: lstm

lstm:
	cd lib/lstm/highway_lstm_cuda; ./make.sh