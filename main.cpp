#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>

using namespace cv;
using namespace std;

//limit value of histogram
int histSize = 256;

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

int main(int argc, char** argv )
{

    ofstream myfile;
    myfile.open ("values.txt");

    //Initializing mat
    Mat image_src;
    Mat eq_histogram_graph( 700, 400, CV_8UC3, Scalar( 255,255,255) );
    Mat histogram_graph( 700, 400, CV_8UC3, Scalar( 255,255,255) );

    //reading image
    image_src = imread( argv[1], 0); // 0, grayscale  >0, color

    //writing grayscale image
    imwrite("image_grayscale.jpg", image_src);

    //vector for histogram and eq_histogram
    vector<int> histogram(256) ;
    vector<int> eq_histogram(256);

    //Getting and drawing histogram
    get_histogram( image_src, histogram_graph, histogram);

    
    //mat of destiny
    Mat image_dest = imread("image_grayscale.jpg", 0);

    get_eqhistogram( image_src, image_dest, eq_histogram_graph, histogram, eq_histogram );
   
    //Updating eq image
    for( int i =  0; i < image_src.rows; i++){
        for( int j = 0; j < image_src.cols; j++){
            //cout << (int) image_dest.at<uchar>(i,j) <<"//" <<  eq_histogram[ (int) image_dest.at<uchar>(i,j) ] << "\t";
            image_dest.at<uchar>(i,j) = (uchar) eq_histogram[ (int) image_dest.at<uchar>(i,j) ];
            //cout << (int) image_dest.at<uchar>(i,j) << endl;
        }
    }
    cout << endl;

    //Writing eq image
    imwrite("image_eq.jpg", image_dest);
    

    //Smooth Filter 
    Mat img_padded = imread("image_grayscale.jpg", 0);
    //Mat img_padded = imread("image_eq.jpg", 0);

    smooth_filter( image_src, img_padded);

    //Writing smoothed image
    imwrite("image_padded.jpg", img_padded);    


    //Applying threshold
    Mat img_threshold = imread("image_padded.jpg", 0);
    //Mat img_threshold = imread("image_grayscale.jpg", 0);
 
    // thresh value
    int thresh_value = 150;

    threshold( img_threshold, thresh_value);

    //Writing threshold image
    imwrite("image_threshold.jpg", img_threshold);


    //Median Filter
    Mat img_median = imread("image_padded.jpg", 0);
    //Mat img_median = imread("image_grayscale.jpg", 0);

    median(img_median);

    myfile.close();

    //Display
    namedWindow("Source ", CV_WINDOW_AUTOSIZE );
    imshow("Source ", image_src );

    namedWindow("calcHist ", CV_WINDOW_AUTOSIZE );
    imshow("calcHist ", histogram_graph );

    namedWindow("EQ calcHist ", CV_WINDOW_AUTOSIZE );
    imshow("EQ calcHist ", eq_histogram_graph );

    namedWindow("Destiny ", CV_WINDOW_AUTOSIZE );
    imshow("Destiny ", image_dest );  

    namedWindow("Padded ", CV_WINDOW_AUTOSIZE );
    imshow("Padded ", img_padded );

    namedWindow("Threshold ", CV_WINDOW_AUTOSIZE );
    imshow("Threshold ", img_threshold );

    namedWindow("Median ", CV_WINDOW_AUTOSIZE );
    imshow("Median ", img_median );

    waitKey(0);

    return 0;
}