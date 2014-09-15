#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    //string path_image = "koala.jpg";

    int histSize = 256;

    //reading image
    Mat image_src, histogram_graph( 700, 700, CV_8UC3, Scalar( 255,255,255) );

    image_src = imread( argv[1], 0); // 0, grayscale  >0, color

    vector<int> histogram(256);
    cout << image_src.rows << "\t" << image_src.cols << endl;
//    cout << (int)image_src.at<uchar> ( 0, 10 ) << endl;


    for( int i = 1 ; i < image_src.rows; i++)
        for( int j = 1 ; j < image_src.cols; j++)
            histogram [ (int) image_src.at<uchar>(i,j) ]++;

    int higher = -1;

    for( vector<int>::iterator it = histogram.begin(); it != histogram.end(); it++){
        if( higher < (*it) ) higher = (*it);
//        cout << (*it) << "\t";
    }

    cout << higher << endl;

    for( vector<int>::iterator it = histogram.begin(); it != histogram.end(); it++)
        cout << (*it) << "\t";

    for( int i = 0 ; i < histSize; i++) {
        cout << " val " << i << "\t hist " << histogram[i] << endl;
        line( histogram_graph, Point( i, 0 ), Point( i, double(histogram[i] / 10) ) , Scalar(0,255,0), 2, 8, 0 );
    }

    cout << endl;

    //line( histogram_graph, Point( 0, 100 ), Point(100, 100 ) , Scalar(0,255,0), 3, 8, 0 );

    /// Display
    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    imshow("calcHist Demo", histogram_graph );

    waitKey(0);

    return 0;
}



