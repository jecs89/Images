#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <math.h>
#include <vector>

using namespace std;
using namespace cv;

string folder_path = "/home/jecs89/git_folder/Images/Base/";

// g++ -o test_validation -L /usr/local/cuda-6.5/lib64 `pkg-config opencv --cflags` test_validation.cpp `pkg-config opencv --libs` -std=c++11

string postfix = "_0.png";

#define spc 6

bool isFind( string s, string pat){
	size_t found = 0;
	found = s.find(pat);

	//cout << found << endl;

	return ( found < 0 || found > s.size() ) ? 0 : 1;
}

int main(){
	
	ifstream name_images    ( folder_path + "name_images.data" );
	ifstream classification ( folder_path + "classification.result");	// File to read image

	vector< pair<int,string> > v_name_images(7200);

	string line1, line2;
	int x, y, size;

	for( int i = 0 ; i < 7200; ++i){
		getline( name_images, line1 );

		v_name_images[i].first  = i;
		v_name_images[i].second = line1;

		//cout << v_name_images[i].first << "\t" << v_name_images[i].second << endl;
	}

	//cout << line1 << endl;
	
	//cout << line1.substr( line1.find("j")+1, line1.find("_") - (line1.find("j")+1) ) << endl;

	ofstream updated_class( folder_path + "updated_class.data");

	for( int i = 0 ; i < 7200 ; ++i ){

		updated_class << setw(spc) << v_name_images[i].first;
		line1 = v_name_images[i].second ;
		updated_class << setw(spc) << line1.substr( line1.find("j")+1, line1.find("_") - (line1.find("j")+1) ) << endl;		
	}
	
	updated_class.close();

	vector<vector<int>> v_classification(100);

	int class_img = 0;
	
	while( !classification.eof() ){
		getline( classification, line1);

		//cout << line1 << endl;
		
		//cout << class_img << endl;
		getline( classification, line1);
		//cout << line1 << endl;

		while( line1.size() > 0){
			line2 = line1.substr( 0, 6 );

			if ( ! (istringstream(line2) >> x) ) x = 0;

			v_classification[class_img].push_back(x);

			//cout << line2 << endl;
			line1 = line1.substr( 6 );
		}

		//cout << v_classification[class_img].size() << endl;

		class_img++;
	}

	ofstream validation( folder_path + "validation.result");

	ifstream reader_class( folder_path+ "updated_class.data");

	double error = 0;

	//vector<vector<int>> true_class(7200);

	
			//cout << v_classification[i][j] << "\t";

	while( !reader_class.eof() ){
		getline( reader_class, line1 );

		if( line1.size() != 0){

			line2 = line1.substr( 0, 6 );

			if ( ! (istringstream(line2) >> x ) ) x = 0;

			line2 = line1.substr( 6, 12 );
			if ( ! (istringstream(line2) >> y ) ) y = 0;
			y = y - 1;

			//cout << x << "\t" << y << endl;

			for( int i = 0 ; i < v_classification.size() ; ++i ){
				for( int j = 0 ; j < v_classification[i].size() ; ++j ){
									
					if( i == x){
						if( v_classification[i][j] != y ) {
							error++;
						}
					}
				}
			}
		}
		//cout << endl << endl;
	}

	cout << (7200-error)/7200 << endl;



	validation.close();

	return(0);
}