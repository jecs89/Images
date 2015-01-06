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

#include "myTypes.h"
#include "myConstants.h"

void displayMat( Mat& img_src, string namewindow );

//print double matrix
void print( matrix& x, int space);

void print( m_int& x, int space);

//check values and update using a thresh
void check_values( double& val, double limit_min, int type);

//copy values of double matrix to Mat
void matrixtoMat( matrix& source, Mat& destiny);

//copy values of double matrix to Mat
void matrixtoMat( m_int& source, Mat& destiny);

//copy values of int matrix to Mat
void Mattomatrix( Mat& source, m_int& destiny);

//copy values of Mat to int vector
void Mattovector( Mat& source, vector<int>& destiny);

//copy values of int vector to Mat
void vectortoMat( vector<int>& source, Mat& destiny );

//copy values of int vector to Mat
void vectortoMat( vector<double>& source, Mat& destiny );

void my_fourier_1d( vector<int>& source, vector<double>& real, vector<double>& imag);

void my_fourier_inv1d( vector<int>& source, vector<double>& real, vector<double>& imag);

//fourier function
void my_fourier( Mat& source, matrix& real, matrix& imag);

//get module of fourier transformation
void get_module_fimage( Mat& destiny, matrix& real, matrix& imag);

//inverse fourier function
void my_fourier_inv( matrix& real, matrix& imag, Mat& destiny);

void get_max( Mat& source, int& max);

int get_median( vector<int> vint);

void get_histogram( Mat image_src, Mat &histogram_graph,  vector<int> &histogram);

void get_eqhistogram( Mat image_src, Mat &image_dest, Mat &eq_histogram_graph, vector<int> histogram, vector<int> &eq_histogram);

void smooth_filter( Mat image_src, Mat &img_padded );

void threshold( Mat &img_threshold, int thresh_value);

void median(Mat &img_median);

void get_average( vector<int> neighborhood, double& val);

void get_superpixels( Mat& source, Mat& destiny, int dim);

void get_values( vector<my_Complex>& source, vector<my_Complex>& destiny, int start, int size, int incr);

void fill_vector( vector<my_Complex>& source, int size);

void fft1d( vector<my_Complex>& source );

void ffti1d( vector<my_Complex>& source );

void MattoComplex(Mat& source, vector<my_Complex>& destiny);

void ComplextoMat( vector<my_Complex>& source, Mat& destiny);

void filter_butterworth( Mat& f_source, Mat& filter, Mat& destiny);

void morph_dilation( m_int& source, m_int& struct_elem );

void morph_erosion( m_int& source, m_int& struct_elem );

void vectomatrix( vector<int>& source, m_int& destiny);

void test_morph_dilation();

void test_morph_erosion();

void get_bordersthreshold( Mat& image_src, Mat& borders);

void initMat( Mat& img_src , int val );

void function_equalization( string path );

void function_sfilters( string path );

void function_tfilters( string path );

void function_borders( string path );

void function_morphological( string path );

void function_segmentation( string path );


void function_dominantcolor( string path );

void dominant_color(Mat& image_src, vector<vector<pair<int,int>>>& v_points, vector<Vec3b>& BGR, int x0, int y0, int x1, int y1);

int eucl_distance( Vec3b p1, Vec3b p2 );

bool isFind( string s, string pat);