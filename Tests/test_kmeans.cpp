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

using namespace cv;
using namespace std;

ofstream my_file("k_means.data"); 

bool compareByvalue(const pair<int,int> &a, const pair<int,int> & b){
    return a.first < b.first ;
}

bool compareByvalue2(const pair<int,int> &a, const pair<int,int> & b){
    return a.second > b.second ;
}


bool compareByvalueInt(const int &a, const int & b){
    return a > b ;
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

void print( vector<pair<int,int>>& v_pair, int size ){
	for( int i = 0 ; i < size ; ++i){
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

	// cout << "initial size : " << v_pair.size() << endl;

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

	// cout << "final size : " << result.size() << endl; 
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

void get_histogram( Mat image_src, Mat &histogram_graph,  vector<int> &histogram){    

	int histSize = 256;
    //dimensions of image
    int image_width = image_src.cols;
    int image_height = image_src.rows;

    cout << "Image's dimension: " << image_src.rows << "\t" << image_src.cols << endl;
    //cout << image_height   << "\t" << image_width    << endl;

    //creation of histogram
    for( int i = 0 ; i < image_src.rows; i++){
        for( int j = 0 ; j < image_src.cols; j++){
            histogram [ (int) image_src.at<uchar>(i,j) ]++;
        }
    }

    int higher = -1;

    //get the maximum value
    for( vector<int>::iterator it = histogram.begin(); it != histogram.end(); it++){
        if( higher < (*it) ) higher = (*it);
//        cout << (*it) << "\t";
    }
    //cout << "Max frequency: " << higher << endl;
/*
    for( vector<int>::iterator it = histogram.begin(); it != histogram.end(); it++)
        cout << (*it) << "\t";
*/

    //Drawing Histogram
    //myfile <<"Histogram \n";
    for( int i = 0 ; i < histSize; i++) {
    //    myfile << " val " << i << "\t hist " << histogram[i] << endl;
        line( histogram_graph, Point( i, 0 ), Point( i, double(histogram[i] / 10) ) , Scalar(0,255,0), 2, 8, 0 );
    }

    //cout << endl;

}

int main(int argc, char** argv ){
    
    Mat image_src = imread( argv[1], 0); // 0, grayscale  >0, color

	imwrite( "grayscale.jpg", image_src );    

    getdim( image_src );	//print dimensions of image

    int k = 5;				//k-means

    int niter = 50;			//# iterations

    int error = 10;		//error

    //Getting histogram 
    Mat histogram_graph( 700, 400, CV_8UC3, Scalar( 255,255,255) );

    //vector for histogram and eq_histogram
    vector<int> histogram(256);
    vector<pair<int,int>> phistogram(256);
    
    //Getting and drawing histogram
    get_histogram( image_src, histogram_graph, histogram);

   	imwrite( "Histogram.jpg", histogram_graph );

   	//sort( histogram.begin(), histogram.end(), compareByvalueInt );

	for( int i = 0 ; i < histogram.size() ; ++i){
   		//if( histogram[i] != 0 ) cout << histogram[i] << endl;
   		phistogram[i] = pair<int,int>( i , histogram[i] );
   		//cout << i << "\t" << histogram[i] << endl;
   	}
   	
   	sort( phistogram.begin(), phistogram.end(), compareByvalue2 );

   	print( phistogram, k );

   	/*//Random positions of k points
   	default_random_engine rng( random_device{}() ); 
    uniform_int_distribution<int> dist( 0, image_src.rows ); 
    //default_random_engine rng( random_device{}() ); 
    uniform_int_distribution<int> dist2( 0, image_src.cols ); 
	*/
    vector<pair<int,int>> kpositions;	//position of k points

    vector< vector<pair<int,int>>> neighborhood( k );	//neighboorhod of k points considering error
    vector< vector<pair<int,int>>> norep_neighborhood;	//neighboorhod without repeated points 

    //Filling kpositions with k points
	/*for( int i = 0 ; i < k ; ++i){
		for( int x = 0; x < image_src.rows; ++x){
				for( int y = 0 ; y < image_src.cols; ++y){
					if( image_src.at<uchar>(x,y) == phistogram[i].first  ){					
						kpositions.push_back( pair<int,int>( x , y ) );
						break;
					}
				}
			break;
		}			
	}*/	

	kpositions.push_back( pair<int,int>( 0 , 0 ) );
	kpositions.push_back( pair<int,int>( 100 , 100 ) );
	kpositions.push_back( pair<int,int>( 400 , 250 ) );


	cout << kpositions.size() << endl;

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
	/*
	for( int i = 0 ; i < k ; ++i){
		cout << mean_values[i] << endl;
	}*/

	timer2 = time(0); 
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;		
	

    Mat image_destiny = imread( argv[1], 0); // 0, grayscale  >0, color


   	timer = time(0); 	

   	int kvalues[] = { 0, 50, 100, 150, 200 };

	for( int ik = 0 ; ik < neighborhood.size() ; ++ik ){		
		for( int jk = 0 ; jk < neighborhood[ik].size(); ++jk){
			for( int x = 0; x < image_src.rows; ++x){
				for( int y = 0 ; y < image_src.cols; ++y){
					// my_file << x << "\t" << neighborhood[ik][jk].first << "\t" << y << "\t" << neighborhood[ik][jk].second << endl;
					if( x == neighborhood[ik][jk].first && y == neighborhood[ik][jk].second ){
						//my_file << x << "\t" << neighborhood[ik][jk].first << "\t" << y << "\t" << neighborhood[ik][jk].second << endl;
						image_destiny.at<uchar>(x,y) = kvalues[ik];//image_destiny.at<uchar>(kpositions[ik].first,kpositions[ik].second);//mean_values[ik];//int(image_src.at<uchar>( norep_neighborhood[ik][jk].first, norep_neighborhood[ik][jk].second ));
					}						
					// else if( x != neighborhood[ik][jk].first && y != neighborhood[ik][jk].second ){
					// 	image_destiny.at<uchar>(x,y) = 255;
					// }
				}
			}
		}
	}	

	my_file.close();

	imwrite( "k_means.jpg", image_destiny );
			
	timer2 = time(0); 
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

	return 0;	
}


