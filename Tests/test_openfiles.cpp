#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
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

string folder_path = "/home/jecs89/git_folder/Images/Base/";

// g++ -o test_openfiles  -L /usr/local/cuda-6.5/lib64 `pkg-config opencv --cflags` test_openfiles.cpp `pkg-config opencv --libs` -std=c++11

string postfix = "_0.png";

bool isFind( string s, string pat){
	size_t found = 0;
	found = s.find(pat);

	//cout << found << endl;

	return ( found < 0 || found > s.size() ) ? 0 : 1;
}

void test_isFind(){
	string n = "hola.jpg";
	cout << isFind(n,postfix) << endl;
}

int main(){
	
	DIR *dpdf;
	struct dirent *epdf;

	dpdf = opendir("./");

	vector<Mat> v_image;

	int cont = 0;
	
	if (dpdf != NULL){
	   while (epdf = readdir(dpdf)){
	   	  string name = epdf->d_name;
	   	  if( isFind(name, postfix)){
	      	printf("Filename: %s %s",epdf->d_name, "\n");
	      
	      	Mat reader = imread( name, 1);
	      	v_image.push_back( reader );
	      	cont ++ ;
	      }	      
	   }
	}

	cout << cont << endl;
	cout << v_image.size() << endl;


	int num_examples = 100;

	//Random positions of k points
   	// default_random_engine rng( random_device{}() ); 
    // uniform_int_distribution<int> dist( 0, v_image.size() );        

	for( int i = 0 ; i < num_examples ; ++i){
		
		string s_tmp = static_cast<ostringstream*>( &(ostringstream() << i ) )->str();		
		if( s_tmp.length() == 1 ){
			s_tmp = "0" + s_tmp ;
		}
		imwrite( folder_path + "Data_" + s_tmp +".png", v_image[ i ] );
	}
	
	return(0);
}