#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <math.h>
#include <vector>
#include <random>

#define space 15

#define FOR(i,n,inc) for( i = -n; i < n; i=i+inc)

using namespace std;
using namespace cv;


__global__ void compare(  ){
	
	int index = threadIdx.x + blockIdx.x*blockDim.x;	
	

}

bool compareByvalue(const pair<int,int> &a, const pair<int,int> & b){
    return a.first < b.first ;
}

void print( Mat& image_src, vector<pair<int,int>>& v_pair ){
	for( int i = 0 ; i < v_pair.size() ; ++i){
		cout << v_pair[i].first << "\t" << v_pair[i].second << "\t" << (int)image_src.at<uchar>(v_pair[i].first, v_pair[i].second) << endl;
	}
}

void print( vector<pair<int,int>>& v_pair ){
	for( int i = 0 ; i < v_pair.size() ; ++i){
		cout << v_pair[i].first << "\t" << v_pair[i].second << endl;
	}
}

void getdim( Mat& image_src ){
	cout << image_src.rows << " x " << image_src.cols << endl;
}

int eucl_dstance( int x1, int x2 ){
	return sqrt( pow(x1-x2,2) );
}

void elim_duplicated( vector<pair<int,int>>& v_pair, vector<pair<int,int>>& result ){

	
	//print( v_pair );
	
	//sort( v_pair.begin(), v_pair.end(), compareByvalue );

	cout << "initial size : " << v_pair.size() << endl;

	//print( v_pair );

	//while(true);

	for( int i = 0 ; i < v_pair.size() - 1 ; ++i ){
		if(! ((v_pair[i].first == v_pair[i+1].first) && (v_pair[i].second == v_pair[i+1].second)) ){
			result.push_back( pair<int,int>( v_pair[i].first, v_pair[i].second) );
		}		
	}

	if(! ((v_pair[v_pair.size()-1].first == v_pair[v_pair.size()-2].first) && (v_pair[v_pair.size()-1].second == v_pair[v_pair.size()-2].second)) ){
			result.push_back( pair<int,int>( v_pair[v_pair.size()-1].first, v_pair[v_pair.size()-1].second) );
	}		

	cout << "final size : " << result.size() << endl; 
}

void test_elimduplicated(){
	vector<pair<int,int>> v_pair;
	vector<pair<int,int>> result;

	for( int i = 0 ; i < 10 ; ++i){
		for( int j = 0 ; j < 10 ; ++j){
			v_pair.push_back( pair<int,int>(i,j));
		}
	}

	print( v_pair );

	elim_duplicated( v_pair, result );

	print( result );
}

int main(void){

	int size_matrix = 8 * ( dim1/incr ) * ( dim2/incr ) * ( dim3/incr );
	cout << "# elements: " << size_matrix << endl;
	
	int size = size_matrix * sizeof(point3D);
	cout << "size of matrix: " << size << endl;

	// Allocate space for device copies of a,b,c
	cudaMalloc( (void **)&d_points, size);
	cudaMalloc( (void **)&d_circles, size_matrix * sizeof(point3D) );
	cudaMalloc( (void **)&d_intersections, size);	

	// Alloc space for host copies of a,b,c and setup input
	points   	  = (point3D *)malloc(size);
	circles 	  = (point3D *)malloc(size_matrix * sizeof(point3D));
	intersections = (point3D *)malloc(size);	

	 Mat image_src = imread( argv[1], 0); // 0, grayscale  >0, color

    getdim( image_src );	//print dimensions of image

    int k = 4;				//k-means

    int niter = 50;			//# iterations

    int error = 1;		//error

   	//Random positions of k points
   	default_random_engine rng( random_device{}() ); 
    uniform_int_distribution<int> dist( 0, image_src.rows ); 
    //default_random_engine rng( random_device{}() ); 
    uniform_int_distribution<int> dist2( 0, image_src.cols ); 

    vector<pair<int,int>> kpositions;	//position of k points

    vector< vector<pair<int,int>>> neighborhood( k );	//neighboorhod of k points considering error
    vector< vector<pair<int,int>>> norep_neighborhood;	//neighboorhod without repeated points 

    //Filling kpositions with k points
	for( int i = 0 ; i < k ; ++i){
		kpositions.push_back( pair<int,int>( dist(rng) , dist2(rng) ) );
	}	

	print( image_src, kpositions );	//print k points and values

    time_t timer = time(0); 

    int cont = 0;

	//for( int i = 0 ; i < niter ; ++i){
		for( int ik = 0 ; ik < kpositions.size() ; ++ik ){

			for( int x = 0; x < image_src.rows; ++x){
				for( int y = 0 ; y < image_src.cols; ++y){			
					//cout << x << "\t" << y << endl;
					
					if( eucl_dstance( int( image_src.at<uchar>(kpositions[ ik ].first,kpositions[ ik ].second)) , int(image_src.at<uchar>(x,y))) < error ){
						//cout << int(image_src.at<uchar>(x,y)) << "\t" << int( image_src.at<uchar>(kpositions[ ik ].first,kpositions[ ik ].second) ) << "\t";
						//cout << eucl_dstance( int( image_src.at<uchar>(kpositions[ ik ].first,kpositions[ ik ].second)) , int(image_src.at<uchar>(x,y))) << endl;
						neighborhood[ ik ].push_back( pair<int,int>(x,y) );
						cont++ ;
					}
				}
			}
		}
	//}

	cout << "total number of points inserted in neighborhood: " <<cont << endl;
	
	//Elimination of repeated points
	for( int ik = 0 ; ik < kpositions.size() ; ++ik ){
		vector<pair<int,int>> result;		
		elim_duplicated( neighborhood[ ik ], result );
		norep_neighborhood.push_back( result );				
	}	

	neighborhood = norep_neighborhood;

	time_t timer2 = time(0); 
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;		
	
	vector<int> mean_values;

	cout << "neighborhood's size " << neighborhood.size() << endl;
	
	timer = time(0); 	
	for( int ik = 0 ; ik < neighborhood.size() ; ++ik ){		
		int sum = 0;
		for( int jk = 0 ; jk < neighborhood[ik].size(); ++jk){
			sum += int(image_src.at<uchar>( neighborhood[ik][jk].first, neighborhood[ik][jk].second ));			
			//cout << sum << endl;
			//cout << "neighborhood" << jk << "'s size " << neighborhood[jk].size() << endl;
		}

		//cout << "neighborhood" << ik << "'s size " << neighborhood[ik].size() << endl;

		mean_values.push_back( sum / neighborhood[ik].size() );
	}
	

	for( int i = 0 ; i < k ; ++i){
		cout << mean_values[i] << endl;
	}

	timer2 = time(0); 
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;		
	

    Mat image_destiny = imread( argv[1], 0); // 0, grayscale  >0, color


   	timer = time(0); 	

	for( int ik = 0 ; ik < neighborhood.size() ; ++ik ){		
		for( int jk = 0 ; jk < neighborhood[ik].size(); ++jk){
			for( int x = 0; x < image_src.rows; ++x){
				for( int y = 0 ; y < image_src.cols; ++y){
					// my_file << x << "\t" << neighborhood[ik][jk].first << "\t" << y << "\t" << neighborhood[ik][jk].second << endl;
					if( x == neighborhood[ik][jk].first && y == neighborhood[ik][jk].second ){
						my_file << x << "\t" << neighborhood[ik][jk].first << "\t" << y << "\t" << neighborhood[ik][jk].second << endl;
						image_destiny.at<uchar>(x,y) = 100;//int(image_src.at<uchar>( norep_neighborhood[ik][jk].first, norep_neighborhood[ik][jk].second ));
					}						
					//else{
					//	image_destiny.at<uchar>(x,y) = 0;
					//}
				}
			}
		}
	}	

	my_file.close();

	imwrite( "k_means.jpg", image_destiny );
			
	timer2 = time(0); 
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;





	int cocient = 1000;
	int num_blocks = index / cocient;
	int num_threadsxblock = cocient; 

	// Copy inputs to device
	cudaMemcpy(d_points, points, size, cudaMemcpyHostToDevice);  // Args: Dir. destino, Dir. origen, tamano de dato, sentido del envio
	cudaMemcpy(d_circles, circles, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_intersections, intersections, size, cudaMemcpyHostToDevice);	

	// Launch add() kernel on GPU	
	add<<< num_blocks, num_threadsxblock>>> (d_points, d_circles, d_intersections);
	
	
	// Copy result back to host
	cudaMemcpy(intersections, d_intersections, size, cudaMemcpyDeviceToHost);

	// Cleanup
	free(points);
	free(circles);
	free(intersections);
	cudaFree(d_points);
	cudaFree(d_circles);
	cudaFree(d_intersections);

	time_t timer2 = time(0);
	cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

	return 0;
}
