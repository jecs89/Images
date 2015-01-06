#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iomanip>
#include <algorithm>
#include <math.h>
#include <vector>

using namespace std;
using namespace cv;

// g++ -o test_houghlines  -L /usr/local/cuda-6.5/lib64 `pkg-config opencv --cflags` test_houghlines.cpp `pkg-config opencv --libs` -std=c++11

int num_lines = 5;

int tolerance = 30;

double PI = 3.1415;

bool linein( int x, int y, double radio, double theta ){

	double val = (-cos(theta)/sin(theta))*x + (radio/sin(theta));

	return ( (int)val == y ) ? 1 : 0 ;
}


int main(int argc, char** argv ){

	Mat img_src  = imread( argv[1], 1);
	Mat img_gray = imread( argv[1], 0);

	double theta = 0;
	double radio = 0;

	int sections = 8;
	int numr = 5;
	double incr = 0.2;

	time_t timer = time(0);

	for( int r = 0 ; r < numr ; ++r){
		for( int t = 0 ; t < 2*PI ; t+= (2*PI/sections)){
			int count = 0;
			for( int i = 0 ; i < img_src.rows ; ++i ){
				for( int j = 0 ; j < img_src.cols ; ++j){
					if( linein( i, j, r, t ) ){

					}
				}
			}
		}
	}

	time_t timer2 = time(0);

	cout <<"Tiempo total: " << difftime(timer2, timer) << endl;		

	return 0;
}
