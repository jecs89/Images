all:
	g++ testandcomparefourier.cpp -std=c++11 -o testandcomparefourier
	g++ -o main -L /usr/local/cuda-6.5/lib64 `pkg-config opencv --cflags` 	main.cpp `pkg-config opencv --libs` -std=c++11 
