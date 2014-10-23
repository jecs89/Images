#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <math.h>

#include "myTypes.h"
#include "myConstants.h"

using namespace cv;
using namespace std;

ofstream my_file("fourier.data"); 

//print double matrix
void print( matrix& x, int space){
    my_file << "Fourier data" << endl;

    double limit_min = 0.00005;

    for(unsigned int row = 0; row < x.size(); row++){
        for(unsigned int col = 0; col < x[0].size(); col++){
                x[row][col] = ( abs(x[row][col]) < limit_min ) ? 0 : x[row][col];
                cout << setw(space) << x[row][col] << " ";

                my_file << setw(space) << x[row][col] << " " ;

        }
        my_file << endl;
        cout << endl;
    }
    cout << endl;
    //my_file.close();
}

void print( m_int& x, int space){
    my_file << "Fourier data" << endl;

    double limit_min = 0.00005;

    for(unsigned int row = 0; row < x.size(); row++){
        for(unsigned int col = 0; col < x[0].size(); col++){
                //x[row][col] = ( abs(x[row][col]) < limit_min ) ? 0 : x[row][col];
                cout << setw(space) << x[row][col] << " ";

                //my_file << setw(space) << x[row][col] << " " ;

        }
        //my_file << endl;
        cout << endl;
    }
    cout << endl;
    //my_file.close();
}


//check values and update using a thresh
void check_values( double& val, double limit_min, int type){
    //double limit_min = 0.00001;
    if( type == 1 ) { val = ( abs(val) < limit_min ) ? 0 : val;}
    else if( type == 2 ) { val = ( abs(val) > limit_min ) ? 255 : val; }
}

//copy values of double matrix to Mat
void matrixtoMat( matrix& source, Mat& destiny){
    
    for( int x = 0; x < destiny.rows; x++){
	   for( int y = 0; y < destiny.cols; y++){
	       destiny.at<uchar>(x,y) = (int) source[x][y];
	   }
    }                  
}

//copy values of double matrix to Mat
void matrixtoMat( m_int& source, Mat& destiny){
    
    for( int x = 0; x < destiny.rows; x++){
       for( int y = 0; y < destiny.cols; y++){
           destiny.at<uchar>(x,y) = (int) source[x][y];
       }
    }                  
}

//copy values of int matrix to Mat
void Mattomatrix( Mat& source, m_int& destiny){
    
    for( int x = 0; x < source.rows; x++){
       for( int y = 0; y < source.cols; y++){
           destiny[x][y] = (int) source.at<uchar>(x,y);
       }
    }                  
}

//copy values of Mat to int vector
void Mattovector( Mat& source, vector<int>& destiny){
    
    for( int x = 0; x < source.rows; x++){
       for( int y = 0; y < source.cols; y++){
           destiny[ x + y ] = int (source.at<uchar>(x,y) );
       }
    }                  
}

//copy values of int vector to Mat
void vectortoMat( vector<int>& source, Mat& destiny ){
 for( int x = 0; x < destiny.rows; x++){
       for( int y = 0; y < destiny.cols; y++){
           destiny.at<uchar>(x,y) = source[ x + y ];
       }
    }                     
}

//copy values of int vector to Mat
void vectortoMat( vector<double>& source, Mat& destiny ){
 for( int x = 0; x < destiny.rows; x++){
       for( int y = 0; y < destiny.cols; y++){
           destiny.at<uchar>(x,y) = int( source[ x + y ] );
       }
    }                     
}


void my_fourier_1d( vector<int>& source, vector<double>& real, vector<double>& imag){
    double theta = 0, sum1 = 0, sum2 = 0;

    int u,x, tam = source.size();

    FOR( u, tam ){
        sum1 = sum2 = 0;
        
        FOR( x, tam ){
            theta = 2 * PI * ( u * x ) / tam ;
            sum1 = sum1 + source[x] * cos(theta);
            sum2 = sum2 - source[x] * sin(theta);
        }   
        real[u] = sum1;
        imag[u] = sum2;
    }
}

void my_fourier_inv1d( vector<int>& source, vector<double>& real, vector<double>& imag){
    double theta = 0, sum1 = 0, sum2 = 0;

    int u,x, tam = source.size();

    FOR( u, tam ){
        sum1 = sum2 = 0;
        
        FOR( x, tam ){
            theta = 2 * PI * ( u * x ) / tam ;
            sum1 = sum1 + source[x] * cos(theta);
            sum2 = sum2 + source[x] * sin(theta);
        }   
        real[u] = sum1/tam;
        imag[u] = sum2/tam;
    }
}


//fourier function
void my_fourier( Mat& source, matrix& real, matrix& imag) { 

    cout << "My fourier" << endl;    
    int mn = source.rows * source.cols;
    double theta = 0, sum1 = 0, sum2 = 0, limit_min = 0.00001;

    int u,v,x,y;

    FOR( u, source.rows){
        FOR( v, source.cols){
            
            sum1 = 0, sum2 = 0;

            FOR( x, source.rows){
                FOR( y, source.cols){
                    //theta = 2 * PI * ( ((u * x) / source.rows) + ((v * y) / source.cols));
	           	    theta = 2 * PI * ( u * x * source.cols + v * y * source.rows) / mn;
		            
                    sum1 = sum1 + (double)source.at<uchar>(x,y) * cos(theta);
                    sum2 = sum2 - (double)source.at<uchar>(x,y) * sin(theta);
                }
            }
		    //cout << "F "<< u << "," << v << "\t" << "sum" << "\t" << sum1 << "\t" << sum2 << endl;
            //check_values( sum1 , limit_min, 1);
            //check_values( sum2 , limit_min, 1);
            real[u][v] = sum1;
            imag[u][v] = sum2;
        }
	}

    //print( real, 10);
    //print( imag, 10);	
}

//get module of fourier transformation
void get_module_fimage( Mat& destiny, matrix& real, matrix& imag){
    my_file << "Module F_Components \n";

    int i, j; double module = 0.0;

    FOR( i, real.size()){
        FOR( j, real[0].size()){
            module = sqrt( real[i][j] * real[i][j] + imag[i][j] * imag[i][j]);
            check_values( module, 0.0, 0 ) ;
            check_values( module, 255.0, 1 ) ;

            my_file << module << setw(7) ;

            destiny.at<uchar>(i,j) = int( module );
        }
        my_file << endl;
    }
    my_file.close();
}

//inverse fourier function
void my_fourier_inv( matrix& real, matrix& imag, Mat& destiny) { 
	
    cout << "My fourierI" << endl;

    int mn = destiny.rows * destiny.cols;
    
    double theta = 0, sum1 = 0, sum2 = 0, limit_min = 0.00001;

    int u,v,x,y;

    FOR( u, destiny.rows){
	    FOR( v, destiny.cols){
            
            sum1 = 0, sum2 = 0;

            FOR( x, destiny.rows){
                FOR( y, destiny.cols){
                    //theta = 2 * PI * ( u * x / destiny.rows + v * y / destiny.cols);
                    theta = 2 * PI * ( u * x * destiny.cols + v * y * destiny.rows) / mn;

                    sum1 = sum1 + real[x][y] * cos(theta);
                    sum2 = sum2 + imag[x][y] * sin(theta);
                }
            }
            //check_values( sum1 , limit_min, 1);
            //check_values( sum2 , limit_min, 1);

            real[u][v] = (double)(sum1/mn);
            imag[u][v] = (double)(sum2/mn);

            destiny.at<uchar>(u,v) = (int)real[u][v] ;
	        //destiny.at<uchar>(u,v) = sum2;
        }
    }

    FOR( x, real.size()){
        FOR( y, real[0].size()){
            real[x][y] = real[x][y] / mn;
            imag[x][y] = imag[x][y] / mn;
        }
    }

    //print( real, 10);
    //print( imag, 10);    
}

void get_max( Mat& source, int& max){
    max = 0;
    for (int i = 0; i < source.rows; i++){        
        for (int j = 0; j < source.cols; j++){
            max = ( source.at<uchar>(i,j) > max) ? source.at<uchar>(i,j) : max;
        }
    }
}

/*
void test_fourier( string name ){

    Mat I = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
    
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).

    namedWindow("Input Image", CV_WINDOW_AUTOSIZE );

    imshow("Input Image"       , I   );    // Show the result

    namedWindow("spectrum magnitude", CV_WINDOW_AUTOSIZE );

    imshow("spectrum magnitude", magI);
}
*/

int get_median( vector<int> vint){
    
    sort( vint.begin(), vint.end() );

    return ( vint.size() % 2 == 0) ? vint[ (vint.size() + 1) / 2 - 1] :  ( vint[ ( vint.size() ) / 2 - 1] + vint[ ( vint.size() + 1) / 2 - 1] ) / 2 ; 
}

void get_histogram( Mat image_src, Mat &histogram_graph,  vector<int> &histogram){    

    //dimensions of image
    int image_width = image_src.cols;
    int image_height = image_src.rows;

    cout << image_src.rows << "\t" << image_src.cols << endl;
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
    cout << higher << endl;
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

    cout << endl;

}

void get_eqhistogram( Mat image_src, Mat &image_dest, Mat &eq_histogram_graph, vector<int> histogram, vector<int> &eq_histogram){

    //cdf vector
    vector<int> cdf(256);

    //Initializing position 0
    eq_histogram[0] = histogram[0];

    int lower = 1000000;

    //calculating cdf
    //myfile <<"\nCDF\n";
    for( int i = 1; i < histogram.size(); i++ ){
        eq_histogram[i] = eq_histogram[i-1] + histogram[i];
        //myfile << " val " << i << "\t = " << eq_histogram[i] << endl;
        if( eq_histogram[i] < lower) lower = eq_histogram[i] ;
    }

    for( int i = 1; i < histogram.size(); i++ ){
        cdf[i] = ( eq_histogram[i]*255 ) / eq_histogram[254] ;
        //cout << cdf [i] << endl;
    }

    cout << "lower : " << lower << endl;

    //calculating eq_histogram
    //myfile <<"\nEQ Table\n";
    for( int i = 0; i < histogram.size(); i++ ){
        eq_histogram[i] = ( (eq_histogram[i] - lower) * 255 )/ ( image_src.rows * image_src.cols - lower);
        //myfile << " val " << i << "\t = " << cdf[i] << endl;
        //cout << "val " << i << " : " <<  eq_histogram[i] << endl;
    }


    //Drawing EQ histogram
    for( int i = 0 ; i < histSize; i++) {
        //cout << " val " << i << "\t hist " << histogram[i] << endl;
        line( eq_histogram_graph, Point( i, 0 ), Point( i, double(eq_histogram[i] ) ) , Scalar(0,255,0), 2, 8, 0 );
    }

}

void smooth_filter( Mat image_src, Mat &img_padded ){

    int image_width = image_src.cols;
    int image_height = image_src.rows;

    cout << "dimensions of padded " << img_padded.rows << "\t" << img_padded.cols << endl;    

    //resizing image
    resize( img_padded, img_padded, Size(), (double)(image_width + 2) /image_width , (double)(image_height + 2)/image_height , INTER_LINEAR );

/*  //To use gauss function
    //constant epsilon 
    const double eps = 2.2204e-16;
    cout << eps << endl;
*/
    cout << "dimensions of padded " << img_padded.rows << "\t" << img_padded.cols << endl;

    //Aplying filter
    for( int i = 1; i < (img_padded.rows - 1); i++ ){
        for( int j = 1; j < (img_padded.cols - 1); j++ ){
  
            //cout << (int)img_padded.at<uchar>(i,j) << "\t";
            img_padded.at<uchar>(i,j)   = (img_padded.at<uchar>(i-1,j-1) * 1/9 ) + (img_padded.at<uchar>(i-1,j) * 1/9 ) + 
                                          (img_padded.at<uchar>(i-1,j+1) * 1/9 ) + (img_padded.at<uchar>(i,j-1) * 1/9 ) +
                                          (img_padded.at<uchar>(i  ,j  ) * 1/9 ) + (img_padded.at<uchar>(i,j+1) * 1/9 ) +
                                          (img_padded.at<uchar>(i+1,j-1) * 1/9 ) + (img_padded.at<uchar>(i+1,j) * 1/9 ) +
                                          (img_padded.at<uchar>(i+1,j+1) * 1/9 ); 

/*          Using mask of second derivative
            img_padded.at<uchar>(i,j)   = (img_padded.at<uchar>(i-1,j-1) * 0 ) + (img_padded.at<uchar>(i-1,j) * 1 ) + 
                                          (img_padded.at<uchar>(i-1,j+1) * 0 ) + (img_padded.at<uchar>(i,j-1) * 1 ) +
                                          (img_padded.at<uchar>(i  ,j  ) * -4 ) + (img_padded.at<uchar>(i,j+1) * 1 ) +
                                          (img_padded.at<uchar>(i+1,j-1) * 0 ) + (img_padded.at<uchar>(i+1,j) * 1 ) +
                                          (img_padded.at<uchar>(i+1,j+1) * 0 ); 
*/
            //cout << (int)img_padded.at<uchar>(i,j) << "\n";
        } 
    }
}

void threshold( Mat &img_threshold, int thresh_value){
    
    for( int i = 0; i < img_threshold.rows; i++){
        for( int j = 0; j < img_threshold.cols; j++){
            img_threshold.at<uchar>(i,j) = ( (int) img_threshold.at<uchar>(i,j) < thresh_value ) ? 0 : 255;
        }
    }
}

void median(Mat &img_median){

    int i = 0 ;
    vector<int> neighborhood(9);
    for( int i = 1; i < (img_median.rows - 1); i++ ){
        for( int j = 1; j < (img_median.cols - 1); j++ ){
             neighborhood[ 0 ] = img_median.at<uchar>(i-1,j-1) ;
             neighborhood[ 1 ] = img_median.at<uchar>(i-1,j  ) ;
             neighborhood[ 2 ] = img_median.at<uchar>(i-1,j+1) ;
             neighborhood[ 3 ] = img_median.at<uchar>(i,j-1  ) ;
             neighborhood[ 4 ] = img_median.at<uchar>(i  ,j  ) ;
             neighborhood[ 5 ] = img_median.at<uchar>(i,j+1  ) ;
             neighborhood[ 6 ] = img_median.at<uchar>(i+1,j-1) ;
             neighborhood[ 7 ] = img_median.at<uchar>(i+1,j  ) ;
             neighborhood[ 8 ] = img_median.at<uchar>(i+1,j+1) ; 

             img_median.at<uchar>(i,j) = get_median(neighborhood);            
        }
    }    
}

void get_average( vector<int> neighborhood, double& val){
    for( int i = 0 ; i < neighborhood.size(); i++){
        val = val + neighborhood[i];
    }
    val = val / neighborhood.size();
}

void get_superpixels( Mat& source, Mat& destiny, int dim){
    
    vector<int> neighborhood( dim * dim ); double val = 0.0; int start = ( dim - 1) / 2;

  
/*
    for( i = 1; i < (source.rows - 1); i = i + 1 ){
        for( j = 1; j < (source.cols - 1); j = j + 1 ){

            neighborhood[ 0 ] = source.at<uchar>(i-1,j-1) ;
            neighborhood[ 1 ] = source.at<uchar>(i-1,j  ) ;
            neighborhood[ 2 ] = source.at<uchar>(i-1,j+1) ;
            neighborhood[ 3 ] = source.at<uchar>(i,j-1  ) ;
            neighborhood[ 4 ] = source.at<uchar>(i  ,j  ) ;
            neighborhood[ 5 ] = source.at<uchar>(i,j+1  ) ;
            neighborhood[ 6 ] = source.at<uchar>(i+1,j-1) ;
            neighborhood[ 7 ] = source.at<uchar>(i+1,j  ) ;
            neighborhood[ 8 ] = source.at<uchar>(i+1,j+1) ;             

            get_average( neighborhood, val);

            if( abs( val - neighborhood[4] ) / neighborhood[4]  < tolerance ){
            destiny.at<uchar>(i-1,j-1) = destiny.at<uchar>(i-1,j  ) = destiny.at<uchar>(i-1,j+1)
            = destiny.at<uchar>(i,j-1  ) = destiny.at<uchar>(i  ,j  ) = destiny.at<uchar>(i,j+1  )
            = destiny.at<uchar>(i+1,j-1)  = destiny.at<uchar>(i+1,j  ) = destiny.at<uchar>(i+1,j+1)
            = (int)val;
            }        
        }                            
    }*/

    cout << neighborhood.size() << endl;


    for( int i = start; i < (source.rows - start); i=i+dim ){
        for( int j = start; j < (source.cols - start ); j=j+dim ){

            int p = 0;
            
            for( int k = i - start ; k <= ( i + start  ); k++){
                for( int l = j - start ; l <= ( j + start ); l++){
                    neighborhood[p] = (int)destiny.at<uchar>(k,l);
                    p++;
                    //cout << p << endl;
                    //cout << "(" << k << "," << l << ")" << "\t";
                }
                
            }
            //cout << " //// " << endl;

            get_average( neighborhood, val);

//            cout << i << "\t" << j << endl;

            for( int k = i - start ; k <= ( i + start ); k++){
                for( int l = j - start ; l <= ( j + start ); l++){
                    destiny.at<uchar>(k,l) = int(val);

  //                  cout << k << "\t" << l << endl;
                }
            }
        }
    }
}

void get_values( vector<my_Complex>& source, vector<my_Complex>& destiny, int start, int size, int incr){
    
    int index , counter = 0;

    for( counter, index = start; counter < size; counter++, index = index + incr ){

        destiny[counter] = ( my_Complex( source[index].real, source[index].imag ) );
    }
}

void fill_vector( vector<my_Complex>& source, int size){
    for( int i = 0 ; i < size; ++i){
        source.push_back( my_Complex(0,0) );
    }
}

void fft1d( vector<my_Complex>& source ){
    
    int N = source.size();

    if ( N <= 1 ) return;

    vector<my_Complex> even;
    vector<my_Complex> odd; 

    fill_vector( even, N/2);
    fill_vector( odd, N/2);

    get_values( source, even, 0, N/2, 2 );
    get_values( source, odd, 1, N/2, 2 );

    fft1d( even );
    fft1d( odd );

    for( int i = 0 ; i < N/2 ; ++i){
        
        double theta = 2 * PI * i / N;
        
        my_Complex th( cos(theta), sin(theta) );

        my_Complex t( th.real*odd[i].real - th.imag * odd[i].imag , th.real *odd[i].imag + th.imag*odd[i].real);

        source[i]       = my_Complex( even[i].real + t.real, even[i].imag + t.imag ) ;

        source[i + N/2] = my_Complex( even[i].real - t.real, even[i].imag - t.imag ) ;
        
    }
}

void ffti1d( vector<my_Complex>& source ){

    for( int i = 0 ; i < source.size() ; ++i){
        source[i].imag = -source[i].imag;
    }

    fft1d( source );

    for( int i = 0 ; i < source.size() ; ++i){
        source[i].imag = -source[i].imag;
    }

    for( int i = 0 ; i < source.size() ; ++i){
        source[i] = my_Complex( source[i].real / source.size() , source[i].imag / source.size() );
    }   
}

void MattoComplex(Mat& source, vector<my_Complex>& destiny){

    for( int x = 0; x < source.rows; x++){
       for( int y = 0; y < source.cols; y++){
           destiny.push_back( my_Complex( int(source.at<uchar>(x,y)), 0 ) );
       }
    }                     

}

void ComplextoMat( vector<my_Complex>& source, Mat& destiny){

    for( int x = 0; x < destiny.rows; x++){
       for( int y = 0; y < destiny.cols; y++){
           //destiny.at<uchar>(x,y) = log( source[x+y].real*source[x+y].real + source[x+y].imag*source[x+y].imag) ;
            destiny.at<uchar>(x,y) = source[x+y].real;
       }
    }                     
}

void filter_butterworth( Mat& f_source, Mat& filter, Mat& destiny){
 
    for( int x = 0; x < destiny.rows; x++){
       for( int y = 0; y < destiny.cols; y++){
           
            destiny.at<uchar>(x,y) = f_source.at<uchar>(x,y) - filter.at<uchar>(x,y);
       }
    }  

}

void morph_dilation( m_int& source, m_int& struct_elem ){

    int dim =  (struct_elem.size() % 2 == 0 ) ? struct_elem.size()/2 : (struct_elem.size() - 1 ) /2;

    m_int copy_source( source.size(), vector<int>(source[0].size()) );

    for( int i = 0 ; i < source.size(); i++){
        for( int j = 0; j < source[0].size(); j++){

            copy_source[i][j] = source[i][j];

        }
    }

    for( int i = 1 ; i < source.size() - 1; i++){
        for( int j = 1; j < source[0].size() - 1; j++){
            
            if( source[i][j] != 0){

                for( int k = 0; k < struct_elem.size(); k++){
                    for( int l = 0 ; l < struct_elem[0].size(); l++){
                        if( struct_elem[k][l] != 0 ){
                            copy_source[ k + i - 1 ] [ l + j - 1] = struct_elem[ k ][ l ];
                        }
                    }
                }
            }                
        }
    }

    for( int i = 0 ; i < source.size(); i++){
        for( int j = 0; j < source[0].size(); j++){

            source[i][j] = copy_source[i][j];

        }
    }

}

void vectomatrix( vector<int>& source, m_int& destiny){
    int size = sqrt( source.size() );

    vector<int> source_inv;
    
    for( int i = 0; i < source.size() ; i++){

        source_inv.push_back( source[ source.size() - i - 1]  );

    }

    //destiny.resize( size, vector<int>(size));

    vector<int> row(size);

    for( int i = 0; i < size; i++){
        row.clear();
        for( int j = 0; j < size; j++){
            row.push_back( source_inv.back() );
            source_inv.pop_back();
        }
        destiny.push_back( row );
    }


    /*for( int i = 0; i < size; ++i){
        for( int j = 0; j < size; ++j){
            destiny[i][j] = source[i*j+1];

            cout << "(" << i <<"," << j << ")" << " " << source[ i + j] << "\t";
        }
        cout << endl;
    }*/
}