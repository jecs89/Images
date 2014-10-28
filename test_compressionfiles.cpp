#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>
#include <bitset>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

typedef vector<int> v_int;
typedef vector< vector<int> > m_int;

struct my_pair{
	int value, index;
	//my_pair(int v, int i){ value = v; index = i;}
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



void dectobin( int number, int& conv ) {
	int s[12],i,b = number;

	string s_num ="", s_tmp="";

	for(i=0; i<=11; i++){
		s[i]=b%2;
		b=b/2;
	}
	

	for(i=0; i<=11; i++){
		
		s_tmp = static_cast<ostringstream*>( &(ostringstream() << s[ 11-i ]) )->str();		

		s_num = s_num + s_tmp;
	} 

	if ( ! (istringstream(s_num) >> conv) ) conv = 0;


	//cout << conv << endl;
}
/*
void dectobin( int number, int& conv, bitset v_bit ) {

	int limit = 12;

	int s[limit],i,b = number;

	string s_num ="", s_tmp="";

	for(i=0; i<=(limit-1) ; i++){
		s[i]=b%2;
		b=b/2;
	}
	

	for(i=0; i<=(limit-1); i++){
		
		s_tmp = static_cast<ostringstream*>( &(ostringstream() << s[ (limit-1)-i ]) )->str();		

		s_num = s_num + s_tmp;
	} 

	if ( ! (istringstream(s_num) >> conv) ) conv = 0;


	//cout << conv << endl;
}
*/
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
	int val = 1024, con = 0;
	for( int i = 0 ; i < val ; i++){
		cout << i << "\t";
		dectobin( i , con );
	}
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

    //Filling v_pair with histogram and indexes
    for( int i = 0 ; i < histogram.size(); ++i){
    	v_pair[i].value = histogram[i] ;
    	v_pair[i].index = i ;
    }

    //Sorting ascendently by index
    sort( v_pair.begin(), v_pair.end(), compareByvalue );
    
    //for( int i = 0 ; i < v_pair.size(); ++i)	cout << v_pair[i].index << "\t" << v_pair[i].value << endl;

   	//Stream to save values
	ofstream my_file("values.txt");

	//count to change values for more frequent from 1  to limit
	int count = 1; int conv = 0;

	Mat tmp = imread( argv[1], 0); // 0, grayscale  >0, color

	for( int i = 0 ; i < v_pair.size(); ++i) my_file << v_pair[i].index << "," << v_pair[i].value << "."; 
	my_file << "$$" << endl;

	for( int k = 0 ; k < histogram.size(); ++k){
		for( int i = 0 ; i < image_src.rows ; ++i ){
			for( int j = 0 ; j < image_src.cols ; ++j ){
				if( image_src.at<uchar>(i,j) == v_pair[k].index ){
					//dectobin( (int) image_src.at<uchar>(i,j), num );
					//my_file << count << "." ;
					tmp.at<uchar>(i,j) = ( count < 256 ) ? count : tmp.at<uchar>(i,j);
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

				bitset<8> b_num;
				//cout << (int)tmp.at<uchar>(i,j) << "\t";
				int pix = (int)tmp.at<uchar>(i,j);

				//cout << pix << "\t";

				dectobin( pix , conv );

				//cout << conv << endl;
				
				string s_num = static_cast<ostringstream*>( &(ostringstream() << conv) )->str();
				
				string s_prefix ="";

				if( s_num.size() < 9 ) {
					for( int i = 0; i < ( 8 - s_num.size()); ++i ){
						s_prefix = s_prefix + "0";
					}
				}

				s_num = s_prefix + s_num;

				//cout << s_num << endl;

				for( int k = 0 ; k < s_num.size(); ++k){

					b_num[k] = ( s_num[k] == 0) ? b_num[k] : b_num[k].set();
				}
 
			    //string mystring =
    			//b_num.to_string<char,string::traits_type,string::allocator_type>();

				for( int k = 0 ; k < b_num.size(); ++k){
					my_file << b_num[k];
				}
				//my_file << s_num ;
				//my_file << (int)tmp.at<uchar>(i,j) << " "; 
		}
		//my_file << endl;
	}

	my_file.close();

	ifstream reader("values.txt") ;

	string line1, line2, line3, line4 ;		
	int num1, num2, num3;

	vector<my_pair> new_table;

	//while( getline( reader, line1 ) ){
	getline( reader, line1 );
	size_t pos = line1.find("$$");

	line2 = line1.substr( 0, pos );
		
	while( line2.size() != 0){
		
		pos = line2.find(".");	num3 = pos;
		//cout << pos << endl;
		line3 = line2.substr( 0 , pos );
		//cout << line2 << endl;			

		pos = line3.find(",");
		line4 = line3.substr( pos+1 );
		line3 = line3.substr( 0, pos );

		//cout << line2 << "\t" << line3 << endl;

		if ( ! (istringstream(line3) >> num1) ) num1 = 0;
		if ( ! (istringstream(line4) >> num2) ) num2 = 0;

		my_pair mp; mp.index = num1; mp.value = num2;
		new_table.push_back( mp );

		//cout << mp.index << "\t" << mp.value << endl;

		line2 = line2.substr( num3 + 1 );
			//cout << line3 << endl;
	}

	getline( reader, line1 );

	//cout << line1 << endl;		
	line2 = line1;
/*
	Mat image_decomp = imread( argv[1], 0); // 0, grayscale  >0, color
	
	//while( line2.size() != 0){
		
		for( int i = 0 ; i < image_decomp.rows ; ++i){
			for( int j = 0 ; j < image_decomp.cols ; ++j){

				if( line2.size() != 0){
					line3 = line2.substr(0, 8 );

					cout << line3 << "\t";

					if ( ! (istringstream(line3) >> num1) ) num1 = 0;

					bintodec( num1, conv );

					image_decomp.at<uchar>(i,j) = conv;

					line2 = line2.substr(8);
				}
			}
		}
	//}

*/

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