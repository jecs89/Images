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

using namespace std;
using namespace cv;

void get_histogram( Mat image_src, Mat &histogram_graph,  vector<int> &histogram){    

    //dimensions of image
    int image_width = image_src.cols;
    int image_height = image_src.rows;

    cout << "Image's dimension: " << image_src.rows << "\t" << image_src.cols << endl;

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
    }

    //Drawing Histogram
    for( int i = 0 ; i < histSize; i++) {
    //    myfile << " val " << i << "\t hist " << histogram[i] << endl;
        line( histogram_graph, Point( i, 0 ), Point( i, double(histogram[i] / 10) ) , Scalar(0,255,0), 2, 8, 0 );
    }
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

bool compareByvalueInt(const int &a, const int & b){
    return a > b ;
}


bool compareByvalue2(const pair<int,int> &a, const pair<int,int> & b){
    return a.second > b.second ;
}

void print( vector<pair<int,int>>& v_pair ){
	for( int i = 0 ; i < v_pair.size() ; ++i){
		cout << v_pair[i].first << "\t" << v_pair[i].second << endl;
	}
}

int diff = 100;
int size = 4;
int tolerance = 5;


void dominant_color(Mat& image_src, vector<vector<pair<int,int>>>& v_points, vector<int>& vR, vector<int>& vG, vector<int>& vB, int x0, int y0, int x1, int y1){

	//cout << x << "\t" << y << endl;
	for( int p = 0 ; p < size ; ++p){
    	for( int q = 0 ; q < v_points[p].size(); ++q){
	    	for( int i = x0 ; i < x1 ; ++i ){
				for( int j = y0 ; j < y1 ; ++j){
					if( i == v_points[p][q].first && j == v_points[p][q].second ){
						image_src.at<Vec3b>(i,j)[0] = vR[p];
						image_src.at<Vec3b>(i,j)[1] = vG[p];
						image_src.at<Vec3b>(i,j)[2] = vB[p];
					}
				}
			}
		}	
	}
}

int main(int argc, char** argv ){
	
	Mat image_src = imread( argv[1], 1); // 0, grayscale  >0, color
	Mat image_src2 = imread( argv[1], 0); // 0, grayscale  >0, color

    Mat histogram_graph( 700, 400, CV_8UC3, Scalar( 255,255,255) );

    vector<int> histogram(256) ;
    vector<pair<int,int>> phistogram(256);

    get_histogram( image_src2, histogram_graph, histogram);

    for( int i = 0 ; i < histogram.size() ; ++i){   		
   		phistogram[i] = pair<int,int>( i , histogram[i] );   		
   	}

    sort( phistogram.begin(), phistogram.end(), compareByvalue2 );

    //print( phistogram );

    vector<int> v_color;
    v_color.push_back( phistogram[0].first );


    int idx = 0;
    for( int i = 1 ; i < histSize ; ++i){
    	//sort(  v_color.begin(), v_color.end() );
    	if( abs(phistogram[i-1].first - phistogram[i].first) > diff && (v_color.size() < size) ){
			v_color.push_back( phistogram[i].first );    		
			cout << "Color : " << phistogram[i].first << "\t" << v_color[idx++] << endl;
    	}
    }    

    cout << "v_color size: " << v_color.size() << endl;

    vector<vector<pair<int,int>>> v_points(size);

    for( int p = 0 ; p < size ; p++){    	
    	for( int i = 0 ; i < image_src2.rows ; ++i ){
			for( int j = 0 ; j < image_src2.cols ; ++j){
				if( ( (int)image_src2.at<uchar>(i,j) - v_color[p] ) < tolerance ){
					v_points[p].push_back( pair<int,int>(i,j) );
					//cout << v_color[p] << endl;
				}
			}
		}
	}	

	vector<int> vR(size);
	vector<int> vG(size);
	vector<int> vB(size);

	for( int i = 0 ; i < size ; ++i){
		vR.push_back( image_src.at<Vec3b>(v_points[i][0].first,v_points[i][0].second)[0]  );
		vG.push_back( image_src.at<Vec3b>(v_points[i][0].first,v_points[i][0].second)[1]  );
		vB.push_back( image_src.at<Vec3b>(v_points[i][0].first,v_points[i][0].second)[2]  );
	}

	int nThreads = thread::hardware_concurrency();
	vector<thread> ths(nThreads);

	cout << ths.size() << endl;

	double incx = double(image_src.rows)/2;
	double incy = double(image_src.cols)/4;

	cout << incx << "\t" << incy << endl;

	vector< pair< pair<int,int> , pair<int,int> > > v_blocks;

	for ( double x = 0 ; x < image_src.rows - 1 ; x+=incx ){
		for( double y = 1 ; y < image_src.cols - 1; y+=incy ){
			cout << int(x) << "\t" << int(y - 1) << "\t--\t" << int(x + double(image_src.rows)/2 )<< "\t" << int( y -1+ double(image_src.cols)/4 )<< endl ;

			v_blocks.push_back( pair<pair<int,int>, pair<int,int>>( 
								pair<int,int>(int(x),int(y - 1)) , pair<int,int>(int(x + double(image_src.rows)/2 ),int( y -1+ double(image_src.cols)/4 )) )  );
		}
	}

	// cout << v_blocks[1].first.first << endl;
	// cout << v_blocks[1].first.second << endl;


    time_t timer = time(0); 

	for ( int x = 0, y = 0, i = 0 ; x < image_src.rows && y < image_src.cols && i < nThreads; x+=( image_src.rows/nThreads ), y+=(image_src.cols/nThreads), i++ ){
		ths[i] = thread( dominant_color, ref(image_src), ref(v_points), ref(vR), ref(vG), ref(vB), v_blocks[i].first.first, v_blocks[i].first.second, v_blocks[i].second.first, v_blocks[i].second.second);
	}
	
	time_t timer2 = time(0); 

	cout <<"Tiempo total: " << difftime(timer2, timer) << endl;		

	timer = time(0); 
	for ( int i = 0; i < nThreads; i++ )
		ths[i].join();
	timer2 = time(0); 
	
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;		
	imwrite( "dominant_color.jpg", image_src);

    timer = time(0); 
	/*
    for( int p = 0 ; p < size ; ++p){
    	for( int q = 0 ; q < v_points[p].size(); ++q){
	    	for( int i = 0 ; i < image_src.rows ; ++i ){
				for( int j = 0 ; j < image_src.cols ; ++j){
					if( i == v_points[p][q].first && j == v_points[p][q].second ){
						image_src.at<Vec3b>(i,j)[0] = vR[p];
						image_src.at<Vec3b>(i,j)[1] = vG[p];
						image_src.at<Vec3b>(i,j)[2] = vB[p];
					}
				}
			}
		}	
	}
	
	timer2 = time(0); 
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;				
*/
	return 0;
}