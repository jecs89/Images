#include <iostream>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

//nvcc -o test_dominantcolorcu test_dominantcolor.cu -std=c++11

struct RGB{
	unsigned int R,G,B;
	/*RGB(unsigned int _R, unsigned int _G, unsigned int _B){
		R = _R;
		G = _G;
		B = _B;
	}*/
};

__global__ void comp(RGB* image_src, pair<int,int>* points, RGB color, int rows, int cols , int nropoints ){

	//printf("%s", "//");	

	int index = threadIdx.x + blockIdx.x*blockDim.x;

	int x = ( (index/rows) < int(index/rows) ) ? int(index/rows) - 1 : (index/rows);

	int y = index % cols;

	//int c = 0;
	for( int i = 0 ; i < nropoints ; ++i){
		if( points[i].first == x && points[i].second == y){
			image_src[index] = color;
		}
	//	c++;
	}

	//printf("%i %c", c,'\n');
}

__global__ void add(int *a, int *b, int *c, int n){
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	c[index] = a[index] + b[index];
}

__global__ void print(int *a){
	if( a[blockIdx.x] != 0)
 		printf("%d \n", blockIdx.x);
}

#define N (1000)
#define M (10000)

int space = 4;

void printRGB(vector<vector<RGB>>& image_src){
	for( int i = 0 ; i < image_src.size() ; ++i ){
    	for ( int j = 0 ; j < image_src[0].size(); ++j){
	   		cout << (int)image_src[i][j].B << "\t" << (int)image_src[i][j].G << "\t" << (int)image_src[i][j].R << endl;
    	}
    }
}

int main(void){

	//Files to read
	ifstream points	("points.data");
	ifstream image  ("image.data");

	if( !points.is_open() || !image.is_open()){
		cout << "Problems opening file \n";
	}

	int ncolors = 8;	// # basic colors

	vector<vector<RGB>> image_src;	//Matrix of image
	vector<vector<pair<int,int>>> v_points(ncolors);	//Matrix of points according to ncolors

	//To read from file
	string line1, line2, sR, sG, sB;
	int  G=0, B=0;

	//Reading dimension of image
	getline( image, line1 );
	line2 = line1;
	line1 = line1.substr(0, space);
	line2 = line2.substr(space);

	if ( ! (istringstream(line1) >> B) ) B = 0;
	if ( ! (istringstream(line2) >> G) ) G = 0;

	//Resizing image_src
	image_src.resize( B );

	for( int i = 0 ; i < B ; ++i)
		image_src[i].resize(G);


    time_t timer = time(0);  

    //Reading image
	while( getline( image, line1 ) ){
		line2 = line1;
		for( int i = 0 ; i < image_src.size(); ++i){
			for( int j = 0 ; j < image_src[0].size(); ++j){

				sB = line1.substr(0,space);
				sG = line1.substr(space,space);
				sR = line1.substr(space*2,space);

				if ( ! (istringstream(sB) >> image_src[i][j].B) ) image_src[i][j].B = 0;
				if ( ! (istringstream(sG) >> image_src[i][j].G) ) image_src[i][j].G = 0;
				if ( ! (istringstream(sR) >> image_src[i][j].R) ) image_src[i][j].R = 0;

				line1 = line1.substr( space*3 );
			}
		}
	}
	image.close();

	time_t timer2 = time(0);
	cout << "Reading image ::: " ;
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

    timer = time(0);
    int i = -1;

    // while( !points.eof() ){    	getline(points,line1);    	cout << line1 << endl; }
    
    //Reading points
    while( !points.eof() ){
    	    	
    	if( line1 == "/////" ){    		
    		i++;
    		//cout << i << endl;
    	}
    	
    	getline( points, line1 );

    	while( line1.size() != 0 && line1 != "/////" ){
    		sB = line1.substr( 0, space );
    		sG = line1.substr( space, space );

    		//cout << sB << "\t" << sG << endl;

    		if ( ! (istringstream(sB) >> B )) B = 0;
			if ( ! (istringstream(sG) >> G )) G = 0;

    		v_points[i].push_back( pair<int,int>( B, G));

    		line1 = line1.substr( space*2 );
    		//cout << line1 << endl;
    	}
    	
    }
    timer2 = time(0);
    cout << "Reading points ::: " ;
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

    //Counting points
    int sum = 0;
    for( int i = 0 ; i < ncolors ; ++i)
    	sum += v_points[i].size();

    cout << sum << endl;

	timer = time(0);

	pair<int,int> *ppoints;	// vector of points
	RGB *pimage_src; // vector with image info
	pair<int,int> *d_points;	// device copy of ppoints
	RGB *d_image_src;	// device copy of pimage_src

	int sizeimag = image_src.size()*image_src[0].size();	// total size of Image
	
	// Allocate space for device copies of a,b,c
	cudaMalloc((void **)&d_points, sum);
	cudaMalloc((void **)&d_image_src, sizeimag);	

	// Alloc space for host copies of a,b,c and setup input
	ppoints = (pair<int,int> *)malloc(sum*sizeof(int)*2);
	pimage_src = (RGB *)malloc(sizeimag);

	//Filling points to ppoints //cout << "Puntos" << endl; 
	int sum2 = 0;
	for(int i = 0; i < ncolors; ++i){
		for( int j = 0 ; j < v_points[i].size() ; ++j){						
			ppoints[i+j] = pair<int,int>( v_points[i][j].first, v_points[i][j].second );
			sum2++;
		}
	}
	cout << sum2 << endl;
	
	//Filling RGB to pimage_src cout << "Image" << endl;
	for(int i = 0; i < image_src.size(); ++i){
		for( int j = 0 ; j < image_src[0].size(); ++j){
			RGB rgb;
			rgb.B = image_src[i][j].B; rgb.G = image_src[i][j].G; rgb.R = image_src[i][j].R;
			pimage_src[i+j] = rgb;
		}
	}

	vector<RGB> v_color;
    
    //8 Basic Colors
    RGB rgb;	
    rgb.B = 0; rgb.G = 0; rgb.R = 0;		v_color.push_back( rgb );
    rgb.B = 0; rgb.G = 0; rgb.R = 255;		v_color.push_back( rgb );
    rgb.B = 0; rgb.G = 255; rgb.R = 0;		v_color.push_back( rgb );
    rgb.B = 0; rgb.G = 255; rgb.R = 255;	v_color.push_back( rgb );
    rgb.B = 255; rgb.G = 0; rgb.R = 0;		v_color.push_back( rgb );
    rgb.B = 255; rgb.G = 0; rgb.R = 255;	v_color.push_back( rgb );
    rgb.B = 255; rgb.G = 255; rgb.R = 255;	v_color.push_back( rgb );
    rgb.B = 255; rgb.G = 255; rgb.R = 255;	v_color.push_back( rgb );
       
	// Copy inputs to device
	cudaMemcpy(d_points, ppoints, sum*sizeof(int)*2, cudaMemcpyHostToDevice);  // Args: Dir. destino, Dir. origen, tamano de dato, sentido del envio
	cudaMemcpy(d_image_src, pimage_src, sizeimag, cudaMemcpyHostToDevice);

	for( int i = 0 ; i < v_color.size(); ++i){
	// Launch add() kernel on GPU
		comp<<<(N+M-1)/M,M>>> ( d_image_src, d_points, v_color[i], image_src.size(), image_src[0].size(), sum );
	}
	
	// Copy result back to host
	cudaMemcpy( pimage_src, d_image_src, sizeimag, cudaMemcpyDeviceToHost);

	timer2 = time(0);
	cout << "Cuda process ::: " ;
	cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

	timer = time(0);
	ofstream image_converted("image_conv.data");
	image_converted << setw(space) << image_src.size() << setw(space) << image_src[0].size() << endl;

	//Writing new Mat image
	for( int i = 0 ; i < image_src.size() ; ++i ){
		for( int j = 0 ; j < image_src[0].size() ; ++j ){
			image_src[i][j] = pimage_src[i+j];
			image_converted << setw(space) << image_src[i][j].B << setw(space) << image_src[i][j].G << setw(space) << image_src[i][j].R ;
		}
	}

	image_converted.close();

	cout << "Writing image ::: " ;
	cout << "Tiempo total: " << difftime(timer2, timer) << endl;
	
	// Cleanup
	free(ppoints);
	free(pimage_src);	
	cudaFree(d_points);
	cudaFree(d_image_src);	
	
	return 0;
}