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

string folder_path = "/home/jecs89/git_folder/Images/Base/";

int space = 6;

//double w_features[] = { 0.2, 0.2, 0.2, 0.2, 0.2, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 1 };

double w_features[] = { 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  0.08};

int n_features = 15;


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


void kmeans( vector<vector<int>>& features, vector<vector<int>>& classification ){

	// for( int i = 0 ; i < features.size() ; ++i ){
	// 	print( features[i] );
	// }

	int tolerance = 1500;
	
	ifstream centroids ( folder_path + "centroids.data");

	string line1, line2; double k, num;

	getline( centroids, line1 );

	if ( ! (istringstream(line1) >> k) ) k = 0;

	cout << k << "Interrup point" << endl;

	vector<vector<double>> k_centroids(k, vector<double>(n_features));

	int ik = 0;
	while( !centroids.eof() ){
		getline( centroids, line1 );

		for( int i = 0 ; i < n_features ; ++i){
			line2 = line1.substr(0,20);
			
			if ( ! (istringstream(line2) >> num) ) num = 0;

			k_centroids[ik][i] = num;
			line1 = line1.substr(20);
		}
		ik++;
	}

	cout << "Comp" << endl;

	int double = 0;
	int niter = 0, limit = 3;
	//while( niter < limit){
	for( int p = 0 ; p < k ; p++){    	
		int c = 0;
    	for( int i = 0 ; i < features.size() ; ++i ){
	
			double e_distance = dist_euclidean( k_centroids[p], features[i] );
	
			if(  e_distance < tolerance ){
				classification[p].push_back( i );
				sum++;
			}

			/*vector<int> tmp(n_features,0);
			for( int kk = 0 ; kk < classification[p].size() ; ++kk){
				for( int nf = 0 ; nf < n_features; ++nf){
					tmp[nf] = tmp[nf] + classification[p][nf];
				}
			}
			for( int nf = 0 ; nf < n_features; ++nf){
				if( classification[p].size() > 0){
					tmp[nf] = tmp[nf]/classification[p].size();
				}
			}
			if( classification[p].size() > 1){
				k_centroids[p] = tmp;
			}*/
		}
	}	
	// niter++;
	// }
	cout << "Classified Images" << sum << endl;
}


int main(){

	ifstream image ( folder_path + "features.data");	// File to read image

	string line1, line2;
	double x, y, size;

	//Reading dimension of features;
	getline( image, line1 );

	//cout << line1 << endl;

	if ( ! (istringstream(line1) >> x) ) x = 0;
	size = x;
	cout << size << endl;
	y = n_features;
	vector<vector<double>> v_features(int(x), vector<int>(y));	

	int c = 0;		
	while( !image.eof() ){
		getline( image, line1 );

		for( int i = 0 ; i < y ; ++i){
			
			line2 = line1.substr(0,20);
			//cout << line2 << "\t";

			if ( ! (istringstream(line2) >> x) ) x = 0;
			//cout << x << "\t";			

			//cout << c << "\t" << i << endl;

			v_features[c][i] = x;

			//cout << v_features[c][i] << "\t";

			line1 = line1.substr(20);
		}
		// cout << endl;
		c++;
	}

	// for( int i = 0 ; i < size ; ++i){
	// 	cout << dist_euclidean( v_features[0], v_features[i] ) << "\t";
	// }

	int k = 100;
	vector<vector<double>> classification(k);

	kmeans( v_features, classification);

	ofstream result( folder_path + "classification.result");

	for( int i = 0 ; i < classification.size(); ++i){
		result << "Class" << i << endl;
		for( int j = 0 ; j < classification[i].size(); ++j){
			result << setw(space) << classification[i][j] ;
		}
		result << endl;
	}

	result.close();

	return 0;
}