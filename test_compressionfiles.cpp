#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

typedef vector<int> v_int;
typedef vector< vector<int> > m_int;

struct my_pair{
	int value, index;
};

bool compareByvalue(const my_pair &a, const my_pair &b){
    return a.value > b.value ;
}

void get_max( v_int& source, int& max, int& index){
    max = 0;
    for (int i = 0; i < source.size() ; i++){               
    	if( source[i] > max ) {
    		max = source[i];
    		index = i;
    	}        
    }
}

void get_maxd( v_int& source, int& max, int& index, int diff){
    max = 0;
    for (int i = 0; i < source.size() ; i++){               
    	if( source[i] > max && source[i] != diff) {
    		max = source[i];
    		index = i;
    	}        
    }
}


void get_histogram( Mat image_src, vector<int> &histogram){    

    //dimensions of image
    int image_width = image_src.cols;
    int image_height = image_src.rows;

    cout << image_src.rows << "\t" << image_src.cols << endl;

    //creation of histogram
    for( int i = 0 ; i < image_src.rows; i++){
        for( int j = 0 ; j < image_src.cols; j++){
            histogram [ (int) image_src.at<uchar>(i,j) ]++;
        }
    }
}


void init_m_int( m_int& table_values, int value){
	for( int i = 0 ; i < table_values.size() ; ++i ){
		for( int j = 0 ; j < table_values.size() ; ++j ){
			table_values[i][j] = value ;
		}
	}
}

void init_v_int( v_int& table_values, int value){
	for( int i = 0 ; i < table_values.size() ; ++i ){		
		table_values[i] = value ;		
	}
}

void print( v_int& table_values ){
	for( int i = 0 ; i < table_values.size() ; ++i ){		
		cout << table_values[i] << endl ;		
	}
}

void print( m_int& table_values ){
	for( int i = 0 ; i < table_values.size() ; ++i ){
		for( int j = 0 ; j < table_values.size() ; ++j ){
			cout << table_values[i][j] << endl ;
		}
	}
}



void dectobin(int number, int& conv) {
	int tmp;

	if(number <= 1) {
		//cout << number;
		return;
	}

	tmp = number%2;
	dectobin( number >> 1, conv );    
	
	conv = tmp;
}

void bintodec( int number, int&conv){
	int tmp = number, count = 0, acum = 0;	
	
	while( tmp != 0 ){

		if( tmp % 10 != 0 ){
			acum = acum + pow( 2, count );
		}

		tmp = tmp / 10;	

		count++;
	}
	conv = acum;	
}

void test_dectobin(){
	int val = 8000, con = 0;
	dectobin( val , con );
}

void test_bintodec(){
	int i = 10101, conv = 0;
	bintodec( i , conv );
	cout << conv;
}

int main(int argc, char** argv ){

	int num = 0, index = 0;

	//Initializing mat & reading image
    Mat image_src = imread( argv[1], 0); // 0, grayscale  >0, color
          
    //vector for histogram and eq_histogram
    vector<int> histogram(256) ;
 
    //Getting and drawing histogram
    get_histogram( image_src, histogram);

    vector<my_pair> v_pair( histogram.size() );

    for( int i = 0 ; i < histogram.size(); ++i){
    	v_pair[i].value = histogram[i] ;
    	v_pair[i].index = i ;
    }

    sort( v_pair.begin(), v_pair.end(), compareByvalue );

/*    for( int i = 0 ; i < histogram.size(); ++i){
    	cout << v_pair[i].value << "," << v_pair[i].index ;
    }
*/

	ofstream my_file("values.txt");

	int count = 1; int conv = 0;

	Mat tmp = imread( argv[1], 0); // 0, grayscale  >0, color

	for( int k = 0 ; k < histogram.size(); ++k){
		for( int i = 0 ; i < image_src.rows ; ++i ){
			for( int j = 0 ; j < image_src.cols ; ++j ){
				if( image_src.at<uchar>(i,j) == v_pair[k].index ){
					//dectobin( (int) image_src.at<uchar>(i,j), num );
					//my_file << count << "." ;
					tmp.at<uchar>(i,j) = count;
				}
				else{
					//my_file << (int) image_src.at<uchar>(i,j) << "." ;
				}
			}
			//my_file << endl;
		}
		count++;
	}


	for( int i = 0 ; i < image_src.rows ; ++i ){
		for( int j = 0 ; j < image_src.cols ; ++j ){
				dectobin( (int)tmp.at<uchar>(i,j), conv );
				my_file << conv ;
		}
		my_file << endl;
	}

	my_file.close();


	//Reading file and making table
/*	ifstream reader("values.txt") ;

	if( !reader.is_open() ){
		cout << "Problems opening file \n" ;
	}

	string line1, line2 ;	
	double x, y, z ;

	int lvl_compression = 4 , number, conv;

	v_int table_values( pow( 2, lvl_compression) ) ;   

	init_v_int( table_values, 0 );

	while( getline( reader, line1 ) ){
		//cout << line1 << endl;
		line2 = line1 ;
		for( int i = 1; i < lvl_compression ; ++i ){		
			
			while( line2.size() > 0){
				
				line2 = line1.substr( 0, i ) ;

				cout << line2 << "\t";

				if ( ! (istringstream(line2) >> number) ) number = 0;

				cout << number << "\t";

				bintodec( number, conv );

				cout << conv << endl;

				table_values[ conv ] ++;

				//binary( number, conv );

				//string s_num = static_cast<ostringstream*>( &(ostringstream() << conv) )->str();

				//table_values[]

				line2 = line2.substr( i, line2.length() ) ;

				//cout << line2 << endl;
			}
		}
	}

	print( table_values );
*/
	
	return 0;
}