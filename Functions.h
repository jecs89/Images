

#include "ProtFunctions.h"

using namespace cv;
using namespace std;

ofstream my_file("fourier.data"); 

void displayMat( Mat& img_src, string namewindow ){
    namedWindow( namewindow, CV_WINDOW_AUTOSIZE );
    imshow( namewindow, img_src );
}

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

    cout << "Image's dimension: " << image_src.rows << "\t" << image_src.cols << endl;
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
    //cout << "Max frequency: " << higher << endl;
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

    //cout << endl;

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

    //cout << "lower : " << lower << endl;

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

    //cout << "dimensions of padded " << img_padded.rows << "\t" << img_padded.cols << endl;    

    //resizing image
    resize( img_padded, img_padded, Size(), (double)(image_width + 2) /image_width , (double)(image_height + 2)/image_height , INTER_LINEAR );

/*  //To use gauss function
    //constant epsilon 
    const double eps = 2.2204e-16;
    cout << eps << endl;
*/
    //cout << "dimensions of padded " << img_padded.rows << "\t" << img_padded.cols << endl;

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

    //cout << neighborhood.size() << endl;


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
            destiny.at<uchar>(x,y) = (int)source[x+y].real;
            //cout << (int)destiny.at<uchar>(x,y) << endl;
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
                            copy_source[ k + i - 1 ] [ l + j - 1] = struct_elem[ k ][ l ]*255;
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

void morph_erosion( m_int& source, m_int& struct_elem ){

    int dim =  (struct_elem.size() % 2 == 0 ) ? struct_elem.size()/2 : (struct_elem.size() - 1 ) /2;

    m_int copy_source( source.size(), vector<int>(source[0].size()) );

    for( int i = 0 ; i < source.size(); i++){
        for( int j = 0; j < source[0].size(); j++){

            copy_source[i][j] = 0;

        }
    }

    for( int i = 1 ; i < source.size() - 1; i++){
        for( int j = 1; j < source[0].size() - 1; j++){
            
            if( source[i][j] != 0){
                int cont = 0;

                for( int k = 0; k < struct_elem.size(); k++){
                    for( int l = 0 ; l < struct_elem[0].size(); l++){
                        if( struct_elem[k][l] == source[i + k - 1][ j + l - 1 ]/255 && struct_elem[k][l] == 1 ){
                            //copy_source[ i + k - 1 ] [ j + l - 1 ] = 255;
                            cont ++;
                        }
                    }
                }
                if ( cont ==  5 ) { copy_source[i][j] = 255; }
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
}

void test_morph_dilation(){
    
    int spc = 5;


    vector<int> v_test = { 0 , 0 , 0 , 0 , 0 , 0 , 0,
                           0 , 0 , 1 , 0 , 1 , 0 , 0,
                           0 , 0 , 0 , 0 , 0 , 0 , 0,
                           0 , 0 , 0 , 0 , 0 , 0 , 0,
                           0 , 0 , 1 , 0 , 1 , 0 , 0,
                           0 , 0 , 0 , 0 , 0 , 0 , 0,
                           0 , 0 , 0 , 0 , 0 , 0 , 0 };

    for( int i = 0 ; i < v_test.size(); i++){
        cout << v_test[i] << " ";
        if( (i+1) % (int)sqrt( v_test.size() ) == 0){
            cout << endl;
        }
    }

    m_int test;

    vectomatrix( v_test, test );
    print( test, spc );

    vector<int> v_struct_elem = { 0, 1, 0,
                                  1, 1, 1,
                                  0, 1, 0};

    m_int struct_elem;

    vectomatrix( v_struct_elem, struct_elem);

    morph_dilation( test, struct_elem );

    print( test, spc );
}

void test_morph_erosion(){
    
    int spc = 5;


    vector<int> v_test = { 0 , 0 , 1 , 0 , 0 , 0 , 0,
                           0 , 1 , 1 , 1 , 1 , 0 , 0,
                           0 , 0 , 1 , 0 , 0 , 0 , 0,
                           0 , 0 , 0 , 1 , 1 , 0 , 0,
                           0 , 0 , 1 , 1 , 1 , 1 , 0,
                           0 , 0 , 0 , 1 , 1 , 0 , 0,
                           0 , 0 , 0 , 0 , 0 , 0 , 0 };

    for( int i = 0 ; i < v_test.size(); i++){
        cout << v_test[i] << " ";
        if( (i+1) % (int)sqrt( v_test.size() ) == 0){
            cout << endl;
        }
    }

    m_int test;

    vectomatrix( v_test, test );
    print( test, spc );

    vector<int> v_struct_elem = { 0, 1, 0,
                                1, 1, 1,
                                0, 1, 0};

    m_int struct_elem;

    vectomatrix( v_struct_elem, struct_elem);

    morph_erosion( test, struct_elem );

    print( test, spc );
}

void get_bordersthreshold( Mat& image_src, Mat& borders){

    for( int x = 0; x < image_src.rows - 1 ; x++){
        for( int y = 0; y < image_src.cols - 1; y++){
            if( image_src.at<uchar>(x,y) != image_src.at<uchar>(x+1,y+1) ){
                borders.at<uchar>(x,y) = image_src.at<uchar>(x,y); 
            }
        }
    }
}

void initMat( Mat& img_src , int val ){
    for( int i = 0 ; i < img_src.rows ; ++i){
        for( int j = 0; j < img_src.cols; ++j){
            img_src.at<uchar>(i,j) = val;
        }
    }

}

void function_equalization( string path ){

    cout << "Equalization\n";

    cout << path << endl;
    
    //Initializing mat
    Mat image_src = imread( path, 0); // 0, grayscale  >0, color

    Mat eq_histogram_graph( 700, 400, CV_8UC3, Scalar( 255,255,255) );
    Mat histogram_graph( 700, 400, CV_8UC3, Scalar( 255,255,255) );

    //reading image

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
    //cout << endl;

    //Writing eq image
    imwrite("image_eq.jpg", image_dest);

    //Display
    displayMat( image_src, "Source Image" );
    displayMat( image_dest, "Eq Image" );
}

void function_sfilters( string path ){

    cout << "Spatial Filters\n";

    Mat image_src = imread( path, 0); // 0, grayscale  >0, color

    //Smooth Filter 
    Mat img_padded = imread(path, 0);
    //Mat img_padded = imread("image_eq.jpg", 0);

    smooth_filter( image_src, img_padded);

    //Writing smoothed image
    imwrite("image_meanfilter.jpg", img_padded);    


    //Applying threshold
    Mat img_threshold = imread("image_meanfilter.jpg", 0);
    //Mat img_threshold = imread("image_grayscale.jpg", 0);
 
    // thresh value
    int thresh_value = 150;
 
    threshold( img_threshold, thresh_value);

    //Writing threshold image
    imwrite("image_threshold.jpg", img_threshold);

    //Median Filter
    Mat img_median = imread("image_meanfilter.jpg", 0);
    //Mat img_median = imread("image_grayscale.jpg", 0);

    median(img_median);

    imwrite("image_medianfilter.jpg", img_threshold);

    //Display
    displayMat( img_padded, "Mean Filter" );
    displayMat( img_median, "Median Filter" );

}

void function_tfilters( string path ){
    cout << "FOURIER" << endl;
    
    //test_fourier(  argv[1] )   ;

    Mat f_source = imread( path, 0);
    Mat f_reverted = imread( path, 0);
    Mat f_butterworth = imread("filter_butterworth.png", 0) ;

    //matrix C_r( f_source.rows, vector<double>(f_source.cols));
    //matrix C_i( f_source.rows, vector<double>(f_source.cols));

    vector<int>  v_source( f_source.rows * f_source.cols);
    vector<double>  v_real( f_source.rows * f_source.cols);
    vector<double>  v_imag( f_source.rows * f_source.cols);

    Mattovector( f_source, v_source );

    time_t timer = time(0); 

    //my_fourier_1d( v_source, v_real, v_imag );

    vector<my_Complex> v_complex;
    MattoComplex( f_source, v_complex );

    fft1d( v_complex );

    //vector<int> magnitude;

    for( int i = 0 ; i < v_complex.size() ; ++i){
        v_complex[i].real =  log2( abs((int)sqrt( powf(v_complex[i].real,2) + powf( v_complex[i].imag,2) )));
       //cout  << v_complex[i].real << endl;
    }

    ComplextoMat( v_complex, f_source);

    for( int i = 0 ; i < f_source.rows ; ++i ){
        for( int j = 0 ; j < f_source.cols ; ++j){
            //cout << (int)f_source.at<uchar>(i,j) << "\t" << v_complex[i+j].real << endl;
        }
    }

    int limx = f_source.rows/2, limy = f_source.cols/2;
    Mat temp = imread( path, 0) ;

    temp = f_source ;

    //cout << temp ;


    for( int i = 0 ; i < limx; ++i){
        for( int j = 0 ; j < limy ; ++j){
            f_source.at<uchar>(i,j) = temp.at<uchar>( limx + i, limy + j);
            f_source.at<uchar>( limx + i, limy + j) = temp.at<uchar>(i,j);
        }
    }

    for( int i = limx ; i < f_source.rows; ++i){
        for( int j = 0 ; j < limy ; ++j){
            f_source.at<uchar>(i,j) = temp.at<uchar>( f_source.rows - i , f_source.rows );
        }
    }


    imwrite( "image_fourier.jpg", f_source);

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

    //Display
    displayMat( f_source, "FF Transform" );
    displayMat( f_reverted, "FFI Transform" );    
}

void function_borders( string path ){

    cout << "Getting borders\n";

    Mat image_dest = imread(path, 0);
    Mat img_threshold = imread(path, 0);
    Mat borders = imread(path, 0);

    initMat(borders, 0);
    
    threshold( img_threshold, 150);

    get_bordersthreshold( img_threshold, borders );

    imwrite( "image_borders.jpg", borders );

    //Display
    displayMat( borders, "Thresh Border" );
}

void function_morphological( string path ){

    cout << "Morpholical Operations\n";

    Mat f_morph = imread( path, 0);
    Mat f_morph2 = imread( path, 0);

    threshold( f_morph, 150);
    threshold( f_morph2, 150);

    imwrite( "image_thresh.jpg", f_morph );

    m_int m_morph( f_morph.rows, vector<int>(f_morph.cols) );

    vector<int> v_struct_elem = { 0, 1, 0,
                                 1, 1, 1,
                                 0, 1, 0};

    m_int struct_elem;

    vectomatrix( v_struct_elem, struct_elem );

    
    Mattomatrix( f_morph, m_morph );

    morph_dilation( m_morph, struct_elem );

    matrixtoMat( m_morph, f_morph );

    imwrite( "image_morph_dilation.jpg", f_morph );
    
    Mattomatrix( f_morph2, m_morph );

    morph_erosion( m_morph, struct_elem );
    matrixtoMat( m_morph, f_morph );

    imwrite( "image_morph_erosion.jpg", f_morph );

    for( int i = 0 ; i < f_morph.rows; i++){
        for( int j = 0 ; j < f_morph.cols; j++){
            f_morph.at<uchar>(i,j) = abs ( (int)f_morph.at<uchar>(i,j) - (int)f_morph2.at<uchar>(i,j) );
        }
    }

    imwrite( "image_contours.jpg", f_morph );    

    //Display
    displayMat( f_morph, "Dilation Operation" );    
    displayMat( f_morph, "Erosion Operation" );

}

void function_segmentation( string path ){

    Mat image_src = imread( path, 0); // 0, grayscale  >0, color


    Mat image_superpixel = imread( path, 0);

    get_superpixels( image_src, image_superpixel, 3);
    imwrite("superpixel_5x5.jpg", image_superpixel);

    for( int i = 0 ; i < 10; i++){
        image_superpixel = imread( "superpixel_5x5.jpg", 0);
        get_superpixels( image_superpixel, image_superpixel, 3);
        imwrite("superpixel_5x5.jpg", image_superpixel);

    }

    Mat image_sum = imread( path, 0);

    for( int i = 0; i < image_sum.rows; i++){
        for( int j = 0; j < image_sum.cols; j++){
        image_sum.at<uchar>(i,j) = int(image_sum.at<uchar>(i,j)) - int( image_superpixel.at<uchar>(i,j) );
        }
    }

    imwrite("image_sum.jpg", image_sum);

    //Display
    displayMat( image_sum, "Superpixel using a mean grid" );    
}