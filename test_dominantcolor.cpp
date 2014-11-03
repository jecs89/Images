#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <math.h>
#include <vector>


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

int diff = 30;
int size = 4;
int tolerance = 10;

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

    vector<vector<pair<int,int>>> v_points;

    for( int p = 0 ; p < size-1 ; p++){
    	cout << "hola" << endl;
    	for( int i = 0 ; i < image_src2.rows ; ++i ){
			for( int j = 0 ; j < image_src2.cols ; ++j){
				//if( ( (int)image_src2.at<uchar>(i,j) - v_color[p] ) < tolerance ){
					v_points[p].push_back( pair<int,int>(i,j) );
					cout << v_color[p] << endl;
				//}
			}
		}
	}	
/*
	vector<int> vR(size);
	vector<int> vG(size);
	vector<int> vB(size);

	for( int i = 0 ; i < size ; ++i){
		vR.push_back( image_src.at<Vec3b>(v_points[i][0].first,v_points[i][0].second)[0]  );
		vG.push_back( image_src.at<Vec3b>(v_points[i][0].first,v_points[i][0].second)[1]  );
		vB.push_back( image_src.at<Vec3b>(v_points[i][0].first,v_points[i][0].second)[2]  );
	}

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

	imwrite( "dominant_color.jpg", image_src);
*/
	return 0;
}