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

#define histSize 256

int size = 8;
int diff = 255;
int space = 4;

using namespace std;
using namespace cv;

// g++ -o test_dominantcolor  -L /usr/local/cuda-6.5/lib64 `pkg-config opencv --cflags` test_dominantcolor.cpp `pkg-config opencv --libs` -std=c++11

void get_histogram( Mat image_src, vector<Mat> &v_histogram_graph,  vector<vector<int>> &histogram){    

	//Initializing histogram
	for( int i = 0 ; i < histogram.size() ; ++i){
		for( int j = 0 ; j < histogram[0].size() ; ++j){
			histogram[i][j] = 0;
		}
	}

    //dimensions of image
    int image_width = image_src.cols;
    int image_height = image_src.rows;

    cout << "Image's dimension: " << image_src.rows << "\t" << image_src.cols << endl;

    //creation of histogram
    for( int i = 0 ; i < image_src.rows; i++){
        for( int j = 0 ; j < image_src.cols; j++){                        
            histogram [0] [int(image_src.at<Vec3b>(i,j)[0])] ++;
			histogram [1] [int(image_src.at<Vec3b>(i,j)[1])] ++;
			histogram [2] [int(image_src.at<Vec3b>(i,j)[2])] ++;						
        }
    }

    //Drawing Histogram
    for( int i = 0 ; i < histSize; i++) {
        line( v_histogram_graph[0], Point( i+10, 0 ), Point( i+10, double(histogram[0][i] / 100) ) , Scalar(255,0,0), 2, 8, 0 );
        line( v_histogram_graph[1], Point( i+10, 0 ), Point( i+10, double(histogram[1][i] / 100) ) , Scalar(0,255,0), 2, 8, 0 );
        line( v_histogram_graph[2], Point( i+10, 0 ), Point( i+10, double(histogram[2][i] / 100) ) , Scalar(0,0,255), 2, 8, 0 );
    }
}

//Image creator of basic colors
void creator_image( int c ){
	Mat image_src( 500, 500, CV_8UC3, Scalar( 0,0,0) ); 

	for( int i = 0 ; i < image_src.rows; i++){
    	for( int j = 0 ; j < image_src.cols; j++){                        
        	image_src.at<Vec3b>(i,j)[c] = 255;
		}
	}
	imwrite( "ExampleBasicColor.jpg", image_src );
}

void easy_dc(Mat image_src){

	int RGB[] = {0,0,0};
	int mn = image_src.rows * image_src.cols ;

	for( int i = 0 ; i < image_src.rows ; ++i ){
		for( int j = 0 ; j < image_src.cols ; ++j){
			RGB[0] += image_src.at<Vec3b>(i,j)[0] ;
			RGB[1] += image_src.at<Vec3b>(i,j)[1] ;
			RGB[2] += image_src.at<Vec3b>(i,j)[2] ;
		}
	}

	cout << "R : " << RGB[0] / mn << endl ; 
	cout << "G : " << RGB[1] / mn << endl ;
	cout << "B : " << RGB[2] / mn << endl ;

}

bool compareByvalueInt(const int &a, const int & b){	return a > b ;	}

bool compareByvalue2(const pair<int,int> &a, const pair<int,int> & b){	return a.second > b.second ;	}

bool compareByvalueVec3b(const Vec3b &a, const Vec3b & b){	return a.val[0] > b.val[1] ;	}

void print( vector<pair<int,int>>& v_pair ){
	for( int i = 0 ; i < v_pair.size() ; ++i){
		cout << v_pair[i].first << "\t" << v_pair[i].second << endl;
	}
}

//Dominant color ready for threads
void dominant_color(Mat& image_src, vector<vector<pair<int,int>>>& v_points, vector<Vec3b>& BGR, int x0, int y0, int x1, int y1){

	for( int p = 0 ; p < size ; ++p){
    	for( int q = 0 ; q < v_points[p].size(); ++q){
	    	for( int i = x0 ; i < x1 ; ++i ){
				for( int j = y0 ; j < y1 ; ++j){
					if( i == v_points[p][q].first && j == v_points[p][q].second ){
						image_src.at<Vec3b>(i,j)[0] = BGR[p].val[0];
						image_src.at<Vec3b>(i,j)[1] = BGR[p].val[1];
						image_src.at<Vec3b>(i,j)[2] = BGR[p].val[2];
					}
				}
			}
		}	
	}
}

void get_histogram_vector( string path){
	//creator_image( 0 );
	
	int channels = 3;

	Mat image_src = imread( path, 1); // 0, grayscale  >0, color

	vector<int> v_image_src(image_src.rows*image_src.cols*3);

	for( int i = 0 ; i < image_src.rows ; ++i ){
		for( int j = 0 ; j < image_src.cols ; ++j){
			v_image_src.push_back(image_src.at<Vec3b>(i,j)[0]) ;
			v_image_src.push_back(image_src.at<Vec3b>(i,j)[1]) ;
			v_image_src.push_back(image_src.at<Vec3b>(i,j)[2]) ;
		}
	}

	vector<vector<int>> histogram2(channels,vector<int>(256));

	for( int i = 0 ; i < v_image_src.size() ; i+=3){
		histogram2[0][v_image_src[i+0]]++;
		histogram2[1][v_image_src[i+1]]++;
		histogram2[2][v_image_src[i+2]]++;
	}

    vector<Mat> v_histogram_graph(channels);

    for( int i = 0 ; i < channels; ++i)
    	v_histogram_graph[i] = Mat( 700, 500, CV_8UC3, Scalar( 255,255,255) );


	for( int i = 0 ; i < histSize; i++) {
        line( v_histogram_graph[0], Point( i+10, 0 ), Point( i+10, double(histogram2[0][i] / 100) ) , Scalar(255,0,0), 2, 8, 0 );
        line( v_histogram_graph[1], Point( i+10, 0 ), Point( i+10, double(histogram2[1][i] / 100) ) , Scalar(0,255,0), 2, 8, 0 );
        line( v_histogram_graph[2], Point( i+10, 0 ), Point( i+10, double(histogram2[2][i] / 100) ) , Scalar(0,0,255), 2, 8, 0 );
    }

    cout << v_histogram_graph[0].size() << endl;

    imwrite( "histogram1.jpg", v_histogram_graph[0]);
    imwrite( "histogram2.jpg", v_histogram_graph[1]);
    imwrite( "histogram3.jpg", v_histogram_graph[2]);

}

int eucl_distance( Vec3b p1, Vec3b p2 ){
	return sqrt( pow( p1.val[0] - p2.val[0], 2) + pow(p1.val[1] - p2.val[1],2) + pow(p1.val[2] - p2.val[2],2) );
}

int main(int argc, char** argv ){

	//Files to save points and Mat image_src
	ofstream my_file("points.data");
	ofstream my_file2("image.data");

	int channels = 3;	// # channels

	Mat image_src = imread( argv[1], 1); // 0, grayscale  >0, color
	Mat image_src2 = imread( argv[1], 1); // 0, grayscale  >0, color
		
    vector<Mat> histogram_graph(channels);	//Mat's vector where histogram's info will be saved

    //Initializing histogram_graph
    for( int i = 0 ; i < channels; ++i)
    	histogram_graph[i] = Mat( 700, 500, CV_8UC3, Scalar( 255,255,255) );

    vector<vector<int>> histogram(channels,vector<int>(256));	//Matrix where histogram will be saved
        
    get_histogram( image_src, histogram_graph, histogram);	//Getting histogram and Mat's with histogram painted

    //Writing RGB Histogram
    imwrite( "histogramB.jpg", histogram_graph[0]);	
    imwrite( "histogramG.jpg", histogram_graph[1]);
    imwrite( "histogramR.jpg", histogram_graph[2]);

    vector<vector<pair<int,int>>> vphistogram( channels, vector<pair<int,int>>(histSize));	//Matrix of index and histogram value

    //Filling vphistogram with index and histogram value
    for( int i = 0 ; i < histogram[0].size() ; ++i){
   		vphistogram[0][i] = pair<int,int>( i , histogram[0][i] );   		
   		vphistogram[1][i] = pair<int,int>( i , histogram[1][i] );   		
   		vphistogram[2][i] = pair<int,int>( i , histogram[2][i] );   		
   	}

   	//Ordering RGB histogram
    sort( vphistogram[0].begin(), vphistogram[0].end(), compareByvalue2 );
    sort( vphistogram[1].begin(), vphistogram[1].end(), compareByvalue2 );
    sort( vphistogram[2].begin(), vphistogram[2].end(), compareByvalue2 );

    vector<Vec3b> v_color;
    
    //8 Basic Colors
    v_color.push_back( Vec3b(0,0,0) );
    v_color.push_back( Vec3b(0,0,255) );
    v_color.push_back( Vec3b(0,255,0) );
    v_color.push_back( Vec3b(0,255,255) );
    v_color.push_back( Vec3b(255,0,0) );
    v_color.push_back( Vec3b(255,0,255) );
    v_color.push_back( Vec3b(255,255,0) );
    v_color.push_back( Vec3b(255,255,255) );

    // cout << "v_color size: " << v_color.size() << endl;

    // for( int i = 0 ; i < v_color.size() ; ++i)
    // 	printf( "%i %i %i %c", v_color[i].val[0], v_color[i].val[1], v_color[i].val[2], '\n' );


    //Matrix with similiar points to colors
    vector<vector<pair<int,int>>> v_points(size);

    int factor = 2;
    //Filling v_points with points near to k colors
    for( int p = 0 ; p < size ; p++){    	
    	for( int i = 0 ; i < image_src2.rows ; ++i ){
			for( int j = 0 ; j < image_src2.cols ; ++j){
				Vec3b p1 = Vec3b( image_src2.at<Vec3b>(i,j)[0],image_src2.at<Vec3b>(i,j)[1],image_src2.at<Vec3b>(i,j)[2]);
				Vec3b p2 = Vec3b( v_color[p].val[0], v_color[p].val[1], v_color[p].val[2] );
				int e_distance = eucl_distance( p1, p2);

				if(  e_distance < diff/factor ){
					v_points[p].push_back( pair<int,int>(i,j) );					
				}
			}
		}
	}
	
	//Writing file with points
	for( int i = 0 ; i < v_points.size() ; ++i){
		my_file << "/////\n" ;
		for( int p = 0 ; p < v_points[i].size() ; ++p){
			my_file << setw(space) << v_points[i][p].first << setw(space) << v_points[i][p].second << setw(space);
		}
		my_file << endl;	
	}

	my_file.close();

	//Writing file with image_src
	my_file2 << setw(space) << image_src.rows << setw(space) << image_src.cols << endl;
	for( int i = 0 ; i < image_src.rows ; ++i){
		for( int j = 0 ; j < image_src.cols ; ++j){
			my_file2 << setw(space) << (int)image_src.at<Vec3b>(i,j)[0] << setw(space) << (int)image_src.at<Vec3b>(i,j)[1] << setw(space) << (int)image_src.at<Vec3b>(i,j)[2];		
		}
	}
	
	my_file2.close();

	// cout << v_points[0].size() << endl;
	// cout << v_points[1].size() << endl;
	// cout << v_points[2].size() << endl;

	cout << "# PIXELES \n";
	cout << image_src2.rows * image_src2.cols << endl;

	int sum	= 0;
	for( int i = 0 ; i < size ; ++i)
		sum += v_points[i].size();
	
	cout << sum << endl;

	//Parallel Process
	int nThreads = thread::hardware_concurrency();	// # threads
	vector<thread> ths(nThreads);	// threads vector

	cout << ths.size() << endl;

	//dimensions of blocks
	double incx = double(image_src.rows)/2;
	double incy = double(image_src.cols)/4;

	cout << incx << "\t" << incy << endl;

	vector< pair< pair<int,int> , pair<int,int> > > v_blocks;	// vector with (x0,y0) - (x1,y1) of blocks corners

	//Filling v_blocks with indexes
	for ( double x = 0 ; x < image_src.rows - 1 ; x+=incx ){
		for( double y = 1 ; y < image_src.cols - 1; y+=incy ){
			cout << int(x) << "\t" << int(y - 1) << "\t--\t" << int(x + double(image_src.rows)/2 )<< "\t" << int( y -1+ double(image_src.cols)/4 )<< endl ;

			v_blocks.push_back( pair<pair<int,int>, pair<int,int>>( 
								pair<int,int>(int(x),int(y - 1)) , pair<int,int>(int(x + double(image_src.rows)/2 ),int( y -1+ double(image_src.cols)/4 )) )  );
		}
	}

    time_t timer = time(0); 

    //Launching threads
	for ( int x = 0, y = 0, i = 0 ; x < image_src.rows && y < image_src.cols && i < nThreads; x+=( image_src.rows/nThreads ), y+=(image_src.cols/nThreads), i++ ){
		ths[i] = thread( dominant_color, ref(image_src), ref(v_points), ref(v_color), v_blocks[i].first.first, v_blocks[i].first.second, v_blocks[i].second.first, v_blocks[i].second.second);
	}
	
	time_t timer2 = time(0); 

	cout <<"Tiempo total: " << difftime(timer2, timer) << endl;		

	timer = time(0); 

	//Joining threads
	for ( int i = 0; i < nThreads; i++ )
		ths[i].join();
	
	timer2 = time(0); 
	
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;		
	imwrite( "dominant_color.jpg", image_src);

	return 0;
}