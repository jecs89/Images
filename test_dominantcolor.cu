#include <iostream>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

struct RGB{
	unsigned char R,G,B;
};

__global__ void add(int *a, int *b, int *c, int n){
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	c[index] = a[index] + b[index];
}

__global__ void print(int *a){
	if( a[blockIdx.x] != 0)
 		printf("%d \n", blockIdx.x);
}

#define N (1024)
#define M (1000000)

int space = 4;

int main(void){

	ifstream points	("points.data");
	ifstream image  ("image.data");

	if( !points.is_open() || !image.is_open()){
		cout << "Problems opening file \n";
	}

	vector<vector<RGB>> image_src;
	vector<pair<int,int>> v_points;

	string line1, line2, sR, sG, sB;
	int R=0, G=0, B=0;

	getline( image, line1 );

	line2 = line1;

	line1 = line1.substr(0, space);
	line2 = line2.substr(space);

	if ( ! (istringstream(line1) >> B) ) B = 0;
	if ( ! (istringstream(line2) >> G) ) G = 0;

	image_src.resize( B );

	for( int i = 0 ; i < B ; ++i)
		image_src[i].resize(G);

    time_t timer = time(0); 

	while( getline( image, line1 ) ){
		line2 = line1;
		for( int i = 0 ; i < image_src.size(); ++i){
			for( int j = 0 ; j < image_src[0].size(); ++j){

				//line1 = line1.substr( 0, space*3 );
				sB = line1.substr(0,space);
				sB = line1.substr(space,space);
				sB = line1.substr(space*2,space);

				if ( ! (istringstream(sB) >> image_src[i][j].B) ) image_src[i][j].B = 0;
				if ( ! (istringstream(sG) >> image_src[i][j].G) ) image_src[i][j].G = 0;
				if ( ! (istringstream(sR) >> image_src[i][j].R) ) image_src[i][j].R = 0;

				line1.substr( space*3 );
			}
		}
	}

	time_t timer2 = time(0);
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

/*	time_t timer = time(0);

	int *a,*b,*c;					// host copies of a,b,c
	int *d_a, *d_b, *d_c;		// device copies of a,b,c
	int size = N * sizeof(int);

	// Allocate space for device copies of a,b,c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Alloc space for host copies of a,b,c and setup input

	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	for(int i=0; i<N; ++i)
	{
		a[i] = i*i;
		b[i] = i*2;
	}

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);  // Args: Dir. destino, Dir. origen, tamano de dato, sentido del envio
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	add<<<(N+M-1)/M,M>>> (d_a, d_b, d_c, N);
	print<<<N,1>>> (d_a);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
/*
	for(int i=0; i<N; ++i)
		std::cout << setw(6) << a[i];

	std::cout << std::endl;

	for(int i=0; i<N; ++i)
		std::cout << setw(6) << b[i];

	std::cout << std::endl;

	for(int i=0; i<N; ++i)
		std::cout << setw(6) << c[i];

	std::cout << std::endl;

	// Cleanup
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	time_t timer2 = time(0);
	cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

*/	return 0;
}