#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <math.h>
#include <vector>
#include <thread>

using namespace std;
using namespace cv;

int space = 4;

double w_features[] = { 0.2, 0.2, 0.2, 0.2, 0.2, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 1 };

// g++ -o test_readingfeatures  -L /usr/local/cuda-6.5/lib64 `pkg-config opencv --cflags` test_readingfeatures.cpp `pkg-config opencv --libs` -std=c++11

double dist_euclidean( vector<int>& feature1, vector<int>& feature2 ){
	double sum = 0;

	for( int i = 0 ; i < feature1.size() ; ++i){
		//sum += w_features[i] * powf( abs( feature1[i] - feature2[i]), 2 );
		sum += powf( abs( feature1[i] - feature2[i]), 2 );
	}
	return sqrt(sum);
}

void print( vector<int>& v){
	for( int i = 0 ; i < v.size() ; ++i ){
		cout << v[i] << "\t";
	}
	cout << endl;
}


void kmeans( vector<vector<int>>& features, int k, vector<vector<int>>& classification ){

	// for( int i = 0 ; i < features.size() ; ++i ){
	// 	print( features[i] );
	// }

	int tolerance = 1000;

	vector<vector<int>> k_centroids(k, vector<int>(19));

	for( int i = 0 ; i < k ; ++i){
		k_centroids[i] = (features[k]);
		//print( k_centroids[i] );
	}

	for( int p = 0 ; p < k ; p++){    	
		int c = 0;
    	for( int i = 0 ; i < features.size() ; ++i ){
	
			double e_distance = dist_euclidean( k_centroids[p], features[i] );
	
			if(  e_distance < tolerance ){
				classification[p].push_back( i );
			}
		}
	}	
}


int main(){

	ifstream image ("dataset_features.data");	// File to read image

	string line1, line2;
	int x, y, size;

	//Reading dimension of features;
	getline( image, line1 );

	//cout << line1 << endl;

	if ( ! (istringstream(line1) >> x) ) x = 0;
	size = x;
	y = 19;
	vector<vector<int>> v_features(x, vector<int>(y));	

	int c = 0;
	while( !image.eof() ){
		getline( image, line1 );		
		for( int i = 0 ; i < y ; ++i){
			
			line2 = line1.substr(0,6);
			//cout << line2 << "\t";

			if ( ! (istringstream(line2) >> x) ) x = 0;
			// cout << x << "\t";
			v_features[c][i] = x;

			// cout << v_features[c][i] << "\t";

			line1 = line1.substr(6);
		}
		// cout << endl;
		c++;
	}

	// for( int i = 0 ; i < size ; ++i){
	// 	cout << dist_euclidean( v_features[0], v_features[i] ) << "\t";
	// }

	int k = 5;
	vector<vector<int>> classification(5);

	kmeans( v_features, k, classification);

	ofstream result("classification.result");

	for( int i = 0 ; i < classification.size(); ++i){
		result << "Class" << i << endl;
		for( int j = 0 ; j < classification[i].size(); ++j){
			result << classification[i][j] << "\t";
		}
		result << endl;
	}

	result.close();

	return 0;
}