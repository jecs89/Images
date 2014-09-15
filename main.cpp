#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    int histSize = 256;

    //reading image
    Mat image_src, histogram_graph( 700, 400, CV_8UC3, Scalar( 255,255,255) ), eq_histogram_graph( 700, 400, CV_8UC3, Scalar( 255,255,255) ), image_dest;

    image_src = imread( argv[1], 0); // 0, grayscale  >0, color
    image_dest = image_src;

    //vector for histogram and eq_histogram
    vector<int> histogram(256) ;
    vector<int> eq_histogram(256);
    vector<int> cdf(256);

//    cout << image_src.rows << "\t" << image_src.cols << endl;

    //creation of histogram
    for( int i = 0 ; i < image_src.rows; i++)
        for( int j = 0 ; j < image_src.cols; j++)
            histogram [ (int) image_src.at<uchar>(i,j) ]++;

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
    for( int i = 0 ; i < histSize; i++) {
        //cout << " val " << i << "\t hist " << histogram[i] << endl;
        line( histogram_graph, Point( i, 0 ), Point( i, double(histogram[i] / 10) ) , Scalar(0,255,0), 2, 8, 0 );
    }

    cout << endl;

    eq_histogram[0] = histogram[0];

    int lower = 1000000;

    //calculating cdf
    for( int i = 1; i < histogram.size(); i++ ){
        eq_histogram[i] = eq_histogram[i-1] + histogram[i];
        if( eq_histogram[i] < lower) lower = eq_histogram[i] ;
    }
    for( int i = 1; i < histogram.size(); i++ ){
        cdf[i] = ( eq_histogram[i]*255 ) / eq_histogram[254] ;
        //cout << cdf [i] << endl;
    }

    cout << "lower : " << lower << endl;

    //calculating eq_histogram
    for( int i = 0; i < histogram.size(); i++ ){
        eq_histogram[i] = ( (eq_histogram[i] - lower) * 255 )/ ( image_src.rows * image_src.cols - lower);
        cout << "val " << i << " : " <<  eq_histogram[i] << endl;
    }


    //Drawing EQ histogram
    for( int i = 0 ; i < histSize; i++) {
        //cout << " val " << i << "\t hist " << histogram[i] << endl;
        line( eq_histogram_graph, Point( i, 0 ), Point( i, double(eq_histogram[i] ) ) , Scalar(0,255,0), 2, 8, 0 );
    }

    //Updating eq image
    for( int i =  0; i < image_src.rows; i++)
        for( int j = 0; j < image_src.cols; j++){
            //cout << (int) image_dest.at<uchar>(i,j) <<"//" <<  eq_histogram[ (int) image_dest.at<uchar>(i,j) ] << "\t";
            image_dest.at<uchar>(i,j) = (uchar) eq_histogram[ (int) image_dest.at<uchar>(i,j) ];
            //cout << (int) image_dest.at<uchar>(i,j) << endl;
        }

    cout << endl;


    //Display
    namedWindow("Source ", CV_WINDOW_AUTOSIZE );
    imshow("Source ", image_src );

    namedWindow("calcHist ", CV_WINDOW_AUTOSIZE );
    imshow("calcHist ", histogram_graph );

    namedWindow("EQ calcHist ", CV_WINDOW_AUTOSIZE );
    imshow("EQ calcHist ", eq_histogram_graph );

    namedWindow("Destiny ", CV_WINDOW_AUTOSIZE );
    imshow("Destiny ", image_dest );

    waitKey(0);

    return 0;
}



