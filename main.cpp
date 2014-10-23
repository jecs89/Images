
#include "Functions.h"

int main(int argc, char** argv ){

    ofstream my_file("values.txt");

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

    //my_file.close();


    cout << "FOURIER" << endl;
    
    //test_fourier(  argv[1] )   ;

    Mat f_source = imread(argv[1], 0);
    Mat f_reverted = imread(argv[1], 0) ;
    Mat f_butterworth = imread("filter_butterworth.png", 0) ;

/*    matrix C_r( f_source.rows, vector<double>(f_source.cols));
    matrix C_i( f_source.rows, vector<double>(f_source.cols));

    vector<int>  v_source( f_source.rows * f_source.cols);
    vector<double>  v_real( f_source.rows * f_source.cols);
    vector<double>  v_imag( f_source.rows * f_source.cols);

    Mattovector( f_source, v_source );
*/
    time_t timer = time(0); 

    //my_fourier_1d( v_source, v_real, v_imag );

    vector<my_Complex> v_complex;
    MattoComplex( f_source, v_complex );

    fft1d( v_complex );

    ComplextoMat( v_complex, f_source);

    imwrite( "imag_fourier.jpg", f_source);

    time_t timer2 = time(0);
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;


    filter_butterworth( f_source, f_butterworth, f_source);

    MattoComplex( f_source, v_complex );


    timer = time(0); 
    //my_fourier( f_source, C_r, C_i );

    ffti1d( v_complex );

    ComplextoMat( v_complex, f_reverted);    

    imwrite( "imag_reverted.jpg", f_reverted);

    timer2 = time(0);
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

    //get_module_fimage( f_reverted, C_r, C_i);
    //imwrite("f_module_image.jpg", f_reverted);
    
    /*matrixtoMat( C_r, f_source);
    imwrite("f_real.jpg", f_source);

    matrixtoMat( C_i, f_source);
    imwrite("f_imag.jpg", f_source);
*/
    /*timer = time(0); 
   //my_fourier_inv( C_r, C_i, f_reverted );

    my_fourier_inv1d( v_source, v_real, v_imag  );


    timer2 = time(0);
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

    /*cout << (int)image_src.at<uchar>(10,10) << "\t" << (int)image_src.at<uchar>(15,15) << "\t" << (int)image_src.at<uchar>(50,100) << endl;

    cout << (int)C_r[10][10] << "\t" << (int)C_r[15][15] << "\t" << C_r[50][100] << endl;

    cout << (int)f_reverted.at<uchar>(10,10) << "\t" << (int)f_reverted.at<uchar>(15,15) << "\t" << (int)f_reverted.at<uchar>(50,100) << endl;
    
    imwrite("f_reverted.jpg", f_reverted);

    matrixtoMat( C_r, f_source);
    imwrite("fi_real.jpg", f_source);

    matrixtoMat( C_i, f_source);
    imwrite("fi_imag.jpg", f_source);

/*    Mat image_superpixel = imread( argv[1], 0);

    get_superpixels( image_src, image_superpixel, 3);
    imwrite("superpixel_5x5.jpg", image_superpixel);

    for( int i = 0 ; i < 10; i++){
        image_superpixel = imread( "superpixel_5x5.jpg", 0);
        get_superpixels( image_superpixel, image_superpixel, 3);
        imwrite("superpixel_5x5.jpg", image_superpixel);

    }

    Mat image_sum = imread( argv[1], 0);

    for( int i = 0; i < image_sum.rows; i++){
        for( int j = 0; j < image_sum.cols; j++){
        image_sum.at<uchar>(i,j) = int(image_sum.at<uchar>(i,j)) - int( image_superpixel.at<uchar>(i,j) );
        }
    }

    imwrite("image_sum.jpg", image_sum);
*/


    //imwrite("f_reverted_imag.jpg", f_imag);

/*
    int my_thresh = 11;

    

    int max_real, max_imag;
    get_max( f_real , max_real);
    get_max( f_imag , max_imag);

    cout << max_real << "\t" << max_imag << endl;

    int c_real = 255 /  log( 1 + max_real ) ;
    int c_imag = 255 /  log( 1 + max_imag ) ;
*/
    /*
    for( int i = 0; i < f_real.rows; i++){
        for( int j = 0; j < f_real.cols; j++){

            f_real.at<uchar>(i,j) = c_real * log( 1 + f_real.at<uchar>(i,j) );
            f_imag.at<uchar>(i,j) = c_imag * log( 1 + f_imag.at<uchar>(i,j) );
            
            f_total.at<uchar>(i,j) =  (1 + sqrt( f_real.at<uchar>(i,j)*f_real.at<uchar>(i,j) + f_imag.at<uchar>(i,j)*f_imag.at<uchar>(i,j) ));
            
            /*cout << (int)f_real.at<uchar>(i,j) << "\t" << (int)f_imag.at<uchar>(i,j) << endl;

            f_real.at<uchar>(i,j) = ( 255/(log(1 + 250))) * log(1 + f_real.at<uchar>(i,j));

            f_imag.at<uchar>(i,j) = ( 255/(log(1 + 250))) * log(1 + f_imag.at<uchar>(i,j));

            //f_total.at<uchar>(i,j) = log(1 + f_real.at<uchar>(i,j)*f_real.at<uchar>(i,j) + f_imag.at<uchar>(i,j)*f_imag.at<uchar>(i,j) );
            //if( f_total.at<uchar>(i,j) < thresh )
            //f_total.at<uchar>(i,j) = ( f_total.at<uchar>(i,j) < my_thresh ) ? 0 : 255;

            cout << (int)f_real.at<uchar>(i,j) << "\t" << (int)f_imag.at<uchar>(i,j) << endl;
        
        }
    }
    */

 /*   namedWindow("My FourierR ", CV_WINDOW_AUTOSIZE );
    imshow("My FourierR ", f_real);

    namedWindow("My FourierI ", CV_WINDOW_AUTOSIZE );
    imshow("My FourierI ", f_imag);

//    namedWindow("Total", CV_WINDOW_AUTOSIZE );
//    imshow("Total", f_total);

    namedWindow("My Fourier ", CV_WINDOW_AUTOSIZE );
    imshow("My Fourier ", f_reverted );
*/

/*
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
*/
    waitKey(0);

    return 0;
}
