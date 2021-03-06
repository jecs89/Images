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

// g++ -o test_readingimages  -L /usr/local/cuda-6.5/lib64 `pkg-config opencv --cflags` test_readingimages.cpp `pkg-config opencv --libs` -std=c++11


int main(){

	ifstream image ("image_conv.data");	// File to read image

	string line1, line2, sR, sG, sB;
	int  G=0, B=0;

	//Reading dimension of image
	getline( image, line1 );
	line2 = line1;
	line1 = line1.substr(0, space);
	line2 = line2.substr(space);

	if ( ! (istringstream(line1) >> B) ) B = 0;
	if ( ! (istringstream(line2) >> G) ) G = 0;

	//Initializing image
	Mat image_src = Mat( B, G, CV_8UC3, Scalar( 255,255,255) );

	for( int i = 0 ; i < image_src.rows; ++i){
		for( int j = 0 ; j < image_src.cols; ++j){
			image_src.at<Vec3b>(i,j)[0] = 255;
			image_src.at<Vec3b>(i,j)[1] = 255;
			image_src.at<Vec3b>(i,j)[2] = 255;
		}
	}
	
	//Filling Mat
	while( getline( image, line1 ) ){
		line2 = line1;
		for( int i = 0 ; i < image_src.rows; ++i){
			for( int j = 0 ; j < image_src.cols; ++j){

				sB = line1.substr(0,space);
				sG = line1.substr(space,space);
				sR = line1.substr(space*2,space);
					
				int iB, iG, iR;
				if ( ! (istringstream(sB) >> iB )) iB = 0;
				if ( ! (istringstream(sG) >> iG )) iG = 0;
				if ( ! (istringstream(sR) >> iR )) iR = 0;

				image_src.at<Vec3b>(i,j)[0] = iB;
				image_src.at<Vec3b>(i,j)[1] = iG;
				image_src.at<Vec3b>(i,j)[2] = iR;
				
				line1 = line1.substr( space*3 );
			}
		}
	}

	imwrite( "DominantColorCuda.jpg", image_src);

	return 0;
}