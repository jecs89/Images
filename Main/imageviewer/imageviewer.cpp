/****************************************************************************
**
** Copyright (C) 2014 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
**     of its contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QtWidgets>
#ifndef QT_NO_PRINTER
#include <QPrintDialog>
#endif

#include "imageviewer.h"

#define PI 3.141593


struct my_Complex{
    double real;
    double imag;

    my_Complex( double _r, double _i){
        real = _r ;
        imag = _i ;
    }
};

string folder_path = "/home/jecs89/qt_projects/imageviewer/";

typedef vector< vector<double> > matrix;
typedef vector< vector<int> > m_int;

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

void Mattomatrix( Mat& source, vector< vector<double> >& destiny){

    for( int x = 0; x < source.rows; x++){
       for( int y = 0; y < source.cols; y++){
           destiny[x][y] = (int) source.at<uchar>(x,y);
       }
    }
}


void fill_vector( vector<my_Complex>& source, int size){
    for( int i = 0 ; i < size; ++i){
        source.push_back( my_Complex(0,0) );
    }
}

//copy values of Mat to int vector
void Mattovector( Mat& source, vector<int>& destiny){
    int p = 0;
    for( int x = 0; x < source.rows; x++){
       for( int y = 0; y < source.cols; y++){
           destiny[ p ] = int (source.at<uchar>(x,y) );
           p++;
       }
    }
    cout << "size vector: " << destiny.size() << "\t" << p << endl;
}

//copy values of int vector to Mat
void vectortoMat( vector<int>& source, Mat& destiny ){
 int p = 0;
 for( int x = 0; x < destiny.rows; x++){
       for( int y = 0; y < destiny.cols; y++){
           destiny.at<uchar>(x,y) = (int)source[ p ];
           p++;
       }
    }
    cout << "size vector: " << source.size() << "\t" << p << endl;
}

//copy values of int vector to Mat
void vectortoMat( vector<double>& source, Mat& destiny ){
    int p = 0;
 for( int x = 0; x < destiny.rows; x++){
       for( int y = 0; y < destiny.cols; y++){
           destiny.at<uchar>(x,y) = int( source[ p ] );
           p++;
       }
    }
     cout << "size vector: " << source.size() << "\t" << p << endl;
}

void get_values( vector<my_Complex>& source, vector<my_Complex>& destiny, int start, int size, int incr){

    int index , counter = 0;

    for( counter, index = start; counter < size; counter++, index = index + incr ){

        destiny[counter] = ( my_Complex( source[index].real, source[index].imag ) );
    }
}

void displayMat( Mat& img_src, string namewindow ){
    namedWindow( namewindow, CV_WINDOW_AUTOSIZE );
    imshow( namewindow, img_src );
}

void threshold( Mat &img_threshold, int thresh_value){

    for( int i = 0; i < img_threshold.rows; i++){
        for( int j = 0; j < img_threshold.cols; j++){
            img_threshold.at<uchar>(i,j) = ( (int) img_threshold.at<uchar>(i,j) < thresh_value ) ? 0 : 255;
        }
    }
}

void morph_dilation( m_int& source, m_int& struct_elem ){

    int dim =  (struct_elem.size() % 2 == 0 ) ? struct_elem.size()/2 : (struct_elem.size() - 1 ) /2;

    m_int copy_source( source.size(), vector<int>(source[0].size()) );

    for( unsigned int i = 0 ; i < source.size(); i++){
        for( unsigned int j = 0; j < source[0].size(); j++){

            copy_source[i][j] = source[i][j];

        }
    }

    for( unsigned int i = 1 ; i < source.size() - 1; i++){
        for( unsigned int j = 1; j < source[0].size() - 1; j++){

            if( source[i][j] != 0){

                for( unsigned int k = 0; k < struct_elem.size(); k++){
                    for( unsigned int l = 0 ; l < struct_elem[0].size(); l++){
                        if( struct_elem[k][l] != 0 ){
                            copy_source[ k + i - 1 ] [ l + j - 1] = struct_elem[ k ][ l ]*255;
                        }
                    }
                }
            }
        }
    }

    for( unsigned int i = 0 ; i < source.size(); i++){
        for( unsigned int j = 0; j < source[0].size(); j++){

            source[i][j] = copy_source[i][j];

        }
    }

}

void morph_erosion( m_int& source, m_int& struct_elem ){

    int dim =  (struct_elem.size() % 2 == 0 ) ? struct_elem.size()/2 : (struct_elem.size() - 1 ) /2;

    m_int copy_source( source.size(), vector<int>(source[0].size()) );

    for( unsigned int i = 0 ; i < source.size(); i++){
        for( unsigned int j = 0; j < source[0].size(); j++){

            copy_source[i][j] = 0;

        }
    }

    for( unsigned int i = 1 ; i < source.size() - 1; i++){
        for( unsigned int j = 1; j < source[0].size() - 1; j++){

            if( source[i][j] != 0){
                int cont = 0;

                for( unsigned int k = 0; k < struct_elem.size(); k++){
                    for( unsigned int l = 0 ; l < struct_elem[0].size(); l++){
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

    for( unsigned int i = 0 ; i < source.size(); i++){
        for( unsigned int j = 0; j < source[0].size(); j++){

            source[i][j] = copy_source[i][j];

        }
    }

}


void vectomatrix( vector<int>& source, m_int& destiny){
    int size = sqrt( source.size() );

    vector<int> source_inv;

    for( unsigned int i = 0; i < source.size() ; i++){

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

//    int test = 10;
//    cout << double(test)/double(N) << " ";

    for( int i = 0 ; i < N/2 ; ++i){

        double theta = -2 * PI * double(i) / N;
        //cout << theta << "\t" ;

        my_Complex th( cos(theta), sin(theta) );

        my_Complex t( th.real*odd[i].real - th.imag * odd[i].imag , th.real *odd[i].imag + th.imag*odd[i].real);

        source[i]       = my_Complex( even[i].real + t.real, even[i].imag + t.imag ) ;

        source[i + N/2] = my_Complex( even[i].real - t.real, even[i].imag - t.imag ) ;

    }
}

void ffti1d( vector<my_Complex>& source ){

    for( unsigned int i = 0 ; i < source.size() ; ++i){
        source[i].imag = -source[i].imag;
    }

    fft1d( source );

    for( unsigned int i = 0 ; i < source.size() ; ++i){
        source[i].imag = -source[i].imag;
    }

    for( unsigned int i = 0 ; i < source.size() ; ++i){
        source[i] = my_Complex( source[i].real / source.size() , source[i].imag / source.size() );
    }
}

void fft2d( vector<my_Complex>& source, int rows, int cols ){

    my_Complex** m_source;

    m_source = (my_Complex**)malloc( rows * sizeof(my_Complex*) );

    for( int i = 0; i < 4; ++i){
        m_source[i] = (my_Complex*) malloc( cols * sizeof(my_Complex ) );
    }

    int p = 0;
    for( int i = 0; i < rows; ++i){
        for( int j = 0; j < cols; ++j){
            m_source[i][j] = source[ p ];
            cout << p << endl;
            p++;
        }
    }

    //Rows
    for( int i = 0; i < rows; ++i){
        my_Complex* m_row;
        m_row = (my_Complex*) malloc( cols * sizeof(my_Complex ) );
        for( int j = 0; j < cols; ++j){
            m_row[j] = m_source[i][j];

        }
        vector<my_Complex> v_row;
        for(int k = 0; k < cols; ++k){
            v_row.push_back( m_row[k] );
        }
        fft1d( v_row );

        for( int j = 0; j < cols; ++j){
            m_source[i][j] = v_row[j];

        }
    }

    //Rows
    for( int i = 0; i < cols; ++i){
        my_Complex* m_row;
        m_row = (my_Complex*) malloc( rows * sizeof(my_Complex ) );
        for( int j = 0; j < rows; ++j){
            m_row[j] = m_source[j][i];

        }
        vector<my_Complex> v_row;
        for(int k = 0; k < rows; ++k){
            v_row.push_back( m_row[k] );
        }

        fft1d( v_row );

        for( int j = 0; j < rows; ++j){
            m_source[j][i] = v_row[j];

        }
    }

    p = 0;
    for( int i = 0; i < rows; ++i){
        for( int j = 0; j < cols; ++j){
            source[p] = m_source[i][j];
            p++;
        }
    }

}

void MattoComplex(Mat& source, vector<my_Complex>& destiny){

    for( int x = 0; x < source.rows; x++){
       for( int y = 0; y < source.cols; y++){
           destiny.push_back( my_Complex( int(source.at<uchar>(x,y)), 0 ) );
       }
    }

    cout << "size of vector: " << destiny.size() << endl;

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

ImageViewer::ImageViewer()
{
    imageLabel = new QLabel;
    imageLabel->setBackgroundRole(QPalette::Base);
    imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    imageLabel->setScaledContents(true);

    scrollArea = new QScrollArea;
    scrollArea->setBackgroundRole(QPalette::Dark);
    scrollArea->setWidget(imageLabel);
    setCentralWidget(scrollArea);

    createActions();
    createMenus();

    resize(QGuiApplication::primaryScreen()->availableSize() * 3 / 5);
}

bool ImageViewer::loadFile( QString &fileName)
{
    //Mat
    sfilename = fileName.toStdString();
    mactual = imread( sfilename, 0); // 0, grayscale  >0, color
    namedWindow( "namewindow", CV_WINDOW_AUTOSIZE );
    imshow( "namewindow", mactual );

    //QImage
    qfilename = fileName;
    QImage image(fileName);
    if (image.isNull()) {
        QMessageBox::information(this, QGuiApplication::applicationDisplayName(),
                                 tr("Cannot load %1.").arg(QDir::toNativeSeparators(fileName)));
        setWindowFilePath(QString());
        imageLabel->setPixmap(QPixmap());
        imageLabel->adjustSize();
        return false;
    }

    qactual = image;

    imageLabel->setPixmap(QPixmap::fromImage(image));

    scaleFactor = 1.0;

    printAct->setEnabled(true);
    fitToWindowAct->setEnabled(true);

    //Activate Buttons
    qHistogram->setEnabled(true);
    qEqHistogram->setEnabled(true);

    qFilter1->setEnabled(true);
    qFilter2->setEnabled(true);
    qFilter3->setEnabled(true);

    qFourier->setEnabled(true);
    qLowPass->setEnabled(true);
    qHighPass->setEnabled(true);

    qErosion->setEnabled(true);
    qDilation->setEnabled(true);

    updateActions();

    if (!fitToWindowAct->isChecked())
        imageLabel->adjustSize();

    setWindowFilePath(fileName);
    return true;
}

void ImageViewer::open()
{
    QStringList mimeTypeFilters;
    foreach (const QByteArray &mimeTypeName, QImageReader::supportedMimeTypes())
        mimeTypeFilters.append(mimeTypeName);
    mimeTypeFilters.sort();
    const QStringList picturesLocations = QStandardPaths::standardLocations(QStandardPaths::PicturesLocation);
    QFileDialog dialog(this, tr("Open File"),
                       picturesLocations.isEmpty() ? QDir::currentPath() : picturesLocations.first());
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setMimeTypeFilters(mimeTypeFilters);
    dialog.selectMimeTypeFilter("image/jpeg");

    while (dialog.exec() == QDialog::Accepted && !loadFile(dialog.selectedFiles().first())) {}
}

void ImageViewer::print()
{
    Q_ASSERT(imageLabel->pixmap());
#if !defined(QT_NO_PRINTER) && !defined(QT_NO_PRINTDIALOG)

    QPrintDialog dialog(&printer, this);

    if (dialog.exec()) {
        QPainter painter(&printer);
        QRect rect = painter.viewport();
        QSize size = imageLabel->pixmap()->size();
        size.scale(rect.size(), Qt::KeepAspectRatio);
        painter.setViewport(rect.x(), rect.y(), size.width(), size.height());
        painter.setWindow(imageLabel->pixmap()->rect());
        painter.drawPixmap(0, 0, *imageLabel->pixmap());
    }
#endif
}

void ImageViewer::zoomIn()
{
    scaleImage(1.25);
}

void ImageViewer::zoomOut()
{
    scaleImage(0.8);
}

void ImageViewer::normalSize()
{
    imageLabel->adjustSize();
    scaleFactor = 1.0;
}

void ImageViewer::fitToWindow()
{
    bool fitToWindow = fitToWindowAct->isChecked();
    scrollArea->setWidgetResizable(fitToWindow);
    if (!fitToWindow) {
        normalSize();
    }
    updateActions();
}

void ImageViewer::about()
{
    QMessageBox::about(this, tr("About Image Viewer"),
            tr("<p>The <b>Image Viewer</b> example shows how to combine QLabel "
               "and QScrollArea to display an image. QLabel is typically used "
               "for displaying a text, but it can also display an image. "
               "QScrollArea provides a scrolling view around another widget. "
               "If the child widget exceeds the size of the frame, QScrollArea "
               "automatically provides scroll bars. </p><p>The example "
               "demonstrates how QLabel's ability to scale its contents "
               "(QLabel::scaledContents), and QScrollArea's ability to "
               "automatically resize its contents "
               "(QScrollArea::widgetResizable), can be used to implement "
               "zooming and scaling features. </p><p>In addition the example "
               "shows how to use QPainter to print an image.</p>"));
}

void ImageViewer:: Histogram(){

    string prefix = "hist";

    int histSize = 256;

    imageLabel->setPixmap(QPixmap::fromImage(qactual));

    //Initializing mat
    Mat image_src = imread( sfilename, 0);

    setWindowFilePath( ImageViewer::qfilename ); // + "@" + image_src.rows + "x" + image_src.cols

    Mat eq_histogram_graph( 700, 400, CV_8UC3, Scalar( 255,255,255) );
    Mat histogram_graph( 700, 400, CV_8UC3, Scalar( 255,255,255) );

    cout << sfilename << endl;

    string true_name = sfilename.substr(23);
    true_name = true_name.substr(0, true_name.find(".") );

    imwrite( folder_path + true_name + "_grayscale.jpg", image_src);

    //vector for histogram and eq_histogram
    vector<int> histogram(256) ;
    vector<int> eq_histogram(256);

    //creation of histogram
    for( int i = 0 ; i < image_src.rows; i++){
        for( int j = 0 ; j < image_src.cols; j++){
            histogram [ (int) image_src.at<uchar>(i,j) ]++;
        }
    }

    //Drawing Histogram
    for( int i = 0 ; i < histSize; i++) {
        line( histogram_graph, Point( i, 0 ), Point( i, double(histogram[i] / 10) ) , Scalar(0,255,0), 2, 8, 0 );
    }

    vector<int> cdf(256);

    //Initializing position 0
    eq_histogram[0] = histogram[0];

    int lower = 1000000;

    //calculating cdf
    for( unsigned int i = 1; i < histogram.size(); i++ ){
        eq_histogram[i] = eq_histogram[i-1] + histogram[i];
        if( eq_histogram[i] < lower) lower = eq_histogram[i] ;
    }

    for( unsigned int i = 1; i < histogram.size(); i++ ){
        cdf[i] = ( eq_histogram[i]*255 ) / eq_histogram[254] ;
    }

    //calculating eq_histogram
    for( unsigned int i = 0; i < histogram.size(); i++ ){
        eq_histogram[i] = ( (eq_histogram[i] - lower) * 255 )/ ( image_src.rows * image_src.cols - lower);
    }

    //Drawing EQ histogram
    for( int i = 0 ; i < histSize; i++) {
        line( eq_histogram_graph, Point( i, 0 ), Point( i, double(eq_histogram[i] ) ) , Scalar(0,255,0), 2, 8, 0 );
    }

    Mat image_dest = imread( sfilename, 0 );

    for( int i =  0; i < image_src.rows; i++){
        for( int j = 0; j < image_src.cols; j++){
            image_dest.at<uchar>(i,j) = (uchar) eq_histogram[ (int) image_dest.at<uchar>(i,j) ];
        }
    }

    namedWindow( "Histogram", CV_WINDOW_AUTOSIZE );
    imshow( "Histogram", histogram_graph );

    namedWindow( "EqHistogram", CV_WINDOW_AUTOSIZE );
    imshow( "EqHistogram", eq_histogram_graph );

    namedWindow( "EqImage", CV_WINDOW_AUTOSIZE );
    imshow( "EqImage", image_dest );

    imwrite( folder_path + true_name + "_histogram.jpg", histogram_graph);
    imwrite( folder_path + true_name + "_eqhistogram.jpg", eq_histogram_graph);
    imwrite( folder_path + true_name + "_eqimage.jpg", image_dest);

}

void ImageViewer:: EqHistogram(){

}

void ImageViewer:: Filter1(){

    string prefix = "sfilter";

    string true_name = sfilename.substr(23);
    true_name = true_name.substr(0, true_name.find(".") );


    Mat image_src = imread( sfilename, 0); // 0, grayscale  >0, color
    Mat img_padded = imread( sfilename, 0); // 0, grayscale  >0, color

    int image_width = image_src.cols;
    int image_height = image_src.rows;

    //cout << "dimensions of padded " << img_padded.rows << "\t" << img_padded.cols << endl;

    //resizing image
    cv::resize( img_padded, img_padded, Size(), (double)(image_width + 2) /image_width , (double)(image_height + 2)/image_height , INTER_LINEAR );

    //Aplying filter
    for( int i = 1; i < (img_padded.rows - 1); i++ ){
        for( int j = 1; j < (img_padded.cols - 1); j++ ){

            img_padded.at<uchar>(i,j)   = (img_padded.at<uchar>(i-1,j-1) * 1/9 ) + (img_padded.at<uchar>(i-1,j) * 1/9 ) +
                                              (img_padded.at<uchar>(i-1,j+1) * 1/9 ) + (img_padded.at<uchar>(i,j-1) * 1/9 ) +
                                              (img_padded.at<uchar>(i  ,j  ) * 1/9 ) + (img_padded.at<uchar>(i,j+1) * 1/9 ) +
                                              (img_padded.at<uchar>(i+1,j-1) * 1/9 ) + (img_padded.at<uchar>(i+1,j) * 1/9 ) +
                                              (img_padded.at<uchar>(i+1,j+1) * 1/9 );
        }
    }

    //Writing smoothed image
    imwrite( folder_path + true_name + "_meanfilter.jpg", img_padded);

    //Applying threshold
    Mat img_threshold = imread( folder_path + true_name + "_meanfilter.jpg", 0);

     // thresh value
    int thresh_value = 150;

    for( int i = 0; i < img_threshold.rows; i++){
        for( int j = 0; j < img_threshold.cols; j++){
            img_threshold.at<uchar>(i,j) = ( (int) img_threshold.at<uchar>(i,j) < thresh_value ) ? 0 : 255;
        }
    }

    //Writing threshold image
    imwrite( folder_path + true_name + "_threshold.jpg", img_threshold);

    namedWindow( "MeanFilter", CV_WINDOW_AUTOSIZE );
    imshow( "MeanFilter", img_padded );
}

void ImageViewer:: Filter2(){

    string prefix = "sfilter";

    string true_name = sfilename.substr(23);
    true_name = true_name.substr(0, true_name.find(".") );

    Mat image_src = imread( sfilename, 0); // 0, grayscale  >0, color
    Mat img_padded = imread( sfilename, 0); // 0, grayscale  >0, color

    int image_width = image_src.cols;
    int image_height = image_src.rows;

    //cout << "dimensions of padded " << img_padded.rows << "\t" << img_padded.cols << endl;

    //resizing image
    cv::resize( img_padded, img_padded, Size(), (double)(image_width + 2) /image_width , (double)(image_height + 2)/image_height , INTER_LINEAR );

    //Aplying filter
    for( int i = 1; i < (img_padded.rows - 1); i++ ){
        for( int j = 1; j < (img_padded.cols - 1); j++ ){


            //Using mask of second derivative
            img_padded.at<uchar>(i,j)   = (img_padded.at<uchar>(i-1,j-1) * 0 ) + (img_padded.at<uchar>(i-1,j) * 1 ) +
                                              (img_padded.at<uchar>(i-1,j+1) * 0 ) + (img_padded.at<uchar>(i,j-1) * 1 ) +
                                              (img_padded.at<uchar>(i  ,j  ) * -4 ) + (img_padded.at<uchar>(i,j+1) * 1 ) +
                                              (img_padded.at<uchar>(i+1,j-1) * 0 ) + (img_padded.at<uchar>(i+1,j) * 1 ) +
                                              (img_padded.at<uchar>(i+1,j+1) * 0 );
        }
    }

    //Writing smoothed image
    imwrite( folder_path + true_name + "_second_derivative.jpg", img_padded);

    imshow( "Second Derivative Mask", img_padded );

}
void ImageViewer:: Filter3(){

    string true_name = sfilename.substr(23);
    true_name = true_name.substr(0, true_name.find(".") );

    //Median Filter
    Mat img_median = imread( sfilename, 0);
    //Mat img_median = imread("image_grayscale.jpg", 0);

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

            sort( neighborhood.begin(), neighborhood.end() );

            int median = ( neighborhood.size() % 2 == 0) ? neighborhood[ (neighborhood.size() + 1) / 2 - 1] :  ( neighborhood[ ( neighborhood.size() ) / 2 - 1] + neighborhood[ ( neighborhood.size() + 1) / 2 - 1] ) / 2  ;

            img_median.at<uchar>(i,j) = median;
        }
    }

    imwrite( folder_path + true_name + "_medianfilter.jpg", img_median);

    imshow( "Median Filter", img_median );

}


void ImageViewer:: Fourier(){

    string true_name = sfilename.substr(23);
    true_name = true_name.substr(0, true_name.find(".") );

    Mat f_source = imread( sfilename, 0);

    vector<int>  v_source( f_source.rows * f_source.cols);
    vector<double>  v_real( f_source.rows * f_source.cols);
    vector<double>  v_imag( f_source.rows * f_source.cols);

    cout << v_source.size() << "\t" << v_real.size() << "\t" << v_imag.size() << endl;

    Mattovector( f_source, v_source );

    //time_t timer = time(0);

    vector<my_Complex> v_complex;
    MattoComplex( f_source, v_complex );

    fft1d( v_complex );

    cout << "End FFT1D\n";

    //fft2d( v_complex, f_source.rows, f_source.cols);


    for( unsigned int i = 0 ; i < v_complex.size() ; ++i){
        v_complex[i].real =   log2( 1 + sqrtf( powf(v_complex[i].real,2) + powf(v_complex[i].imag,2) ));
        //cout  << v_complex[i].real << "\t";
    }

    ComplextoMat( v_complex, f_source);
/*
    for( int i = 0 ; i < f_source.rows ; ++i ){
        for( int j = 0 ; j < f_source.cols ; ++j){
            cout << (int)f_source.at<uchar>(i,j) << "\t" << v_complex[i+j].real << endl;
        }
    }*/

    int limx = f_source.rows/2, limy = f_source.cols/2;
    Mat temp = imread( sfilename, 0) ;

    temp = f_source ;

    //cout << temp ;

/*
    for( int i = 0 ; i < limx; ++i){
        for( int j = 0 ; j < limy ; ++j){
            f_source.at<uchar>(i,j) = temp.at<uchar>( limx + i, limy + j);
            f_source.at<uchar>( limx + i, limy + j) = temp.at<uchar>(i,j);
        }
    }

    int pi = 0, pj = limy;
    for( int i = limx ; i < f_source.rows; ++i){
        for( int j = 0 ; j < limy ; ++j){
            f_source.at<uchar>(i,j) = temp.at<uchar>( pi , pj );

            f_source.at<uchar>(pi,pj) = temp.at<uchar>( i , j );

            pj++;
        }
        pj = limy;
        pi++;
    }
*/
    normalize(f_source, f_source, 0, 1, CV_MINMAX);

    cout << "Writing file\n";

    imwrite( folder_path + true_name + "_fourier.jpg", f_source);

    cout << "End Writing file\n";

}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void ImageViewer:: LowPass(){

    string true_name = sfilename.substr(23);
    true_name = true_name.substr(0, true_name.find(".") );

    cout << true_name << endl;

        Mat I = imread(sfilename, CV_LOAD_IMAGE_GRAYSCALE);

        Mat padded;                            //expand input image to optimal size
        int m = getOptimalDFTSize( I.rows );
        int n = getOptimalDFTSize( I.cols ); // on the border add zero values
        copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

        Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
        Mat complexI;
        merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

        dft(complexI, complexI);            // this way the result may fit in the source matrix

        //displayMat( complexI, "Complex I");

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


        //normalize(magI, magI, 0, 1, CV_MINMAX);
        //displayMat( magI, "Mag I");


        Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        //normalize(q0, q0, 0, 1, CV_MINMAX);
        //displayMat( q0, "q 0");
        Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
        //normalize(q1, q1, 0, 1, CV_MINMAX);
        //displayMat( q1, "q 1");
        Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
        //normalize(q2, q2, 0, 1, CV_MINMAX);
        //displayMat( q2, "q 2");
        Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
        //normalize(q3, q3, 0, 1, CV_MINMAX);
        //displayMat( q3, "q 3");

        Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        //displayMat( tmp, "tmp1");
        q3.copyTo(q0);
        //displayMat( tmp, "tmp2");
        tmp.copyTo(q3);
        //displayMat( tmp, "tmp3");
        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        //displayMat( tmp, "tmp4");
        q2.copyTo(q1);
        //displayMat( tmp, "tmp5");
        tmp.copyTo(q2);
        //displayMat( tmp, "tmp6");

        //normalize(tmp, tmp, 0, 1, CV_MINMAX);

        //displayMat( tmp, "tmpFinal");

        //displayMat( magI, "Mag I2");


        normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                                // viewable image form (float between values 0 and 1).

        //displayMat( magI, "Mag I3");

        Mat myI;

        magI.convertTo(myI, CV_8UC1, 255.0);
/*
        vector< vector<double> > m_myI(magI.rows, vector<double>(magI.cols));

        Mattomatrix( magI,m_myI );

        matrixtoMat( m_myI, myI);


        /*int mayor = 0;

        for( int i = 0; i < 377; ++i){
            for( int j = 0; j < 549; ++j){

                myI.at<uchar>(i,j) = (unsigned int) magI.at<uchar>(i,j);

                //cout << (int) myI.at<uchar>(i,j) << " - ";
                //cout << (int) magI.at<uchar>(i,j) << "\t";

                mayor = (mayor< (int) magI.at<uchar>(i,j) ) ? (int) magI.at<uchar>(i,j) : mayor;
            }
            //cout << ";";
        }

        cout << mayor << endl;

        for( int i = 0; i < 377; ++i){
            for( int j = 0; j < 549; ++j){

                myI.at<uchar>(i,j) = int(myI.at<uchar>(i,j)) / 3 ;

            }
        }
/*
        for( int i = 0; i < myI.rows; ++i){
            for( int j = 0; j < myI.cols; ++j){
                myI.at<uchar>(i,j) = ((int) magI.at<uchar>(i,j) / mayor) * 255 ;
            }
        }

*/
        cout << (int)myI.at<uchar>(190,225) << "\t" << magI.at<uchar>(190,225) << endl;
        cout << magI.type() << endl;

        string ty =  type2str( 5 );
        cout << ty << endl;

        //magI.convertTo( myI, CV_32SC1);
/*
        vector<int> compression_params;
            compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);
*/
        displayMat( myI, "MyI");

        cout << folder_path + true_name + "_fourier.jpg" << endl;

        imwrite( folder_path + true_name + "_fourier.jpg", myI);

        //imshow("Input Image"       , I   );    // Show the result
        imshow("spectrum magnitude", magI);
        waitKey();


}

void ImageViewer:: HighPass(){

}

void ImageViewer:: Erosion(){

    string true_name = sfilename.substr(23);
    true_name = true_name.substr(0, true_name.find(".") );

    Mat f_morph = imread( sfilename, 0);

    m_int m_morph( f_morph.rows, vector<int>(f_morph.cols) );

    Mat f_morph2 = imread( sfilename, 0);
    threshold( f_morph2, 150);

    vector<int> v_struct_elem = { 0, 1, 0,
                                 1, 1, 1,
                                 0, 1, 0};

    m_int struct_elem;

    vectomatrix( v_struct_elem, struct_elem );

    Mattomatrix( f_morph2, m_morph );

    morph_erosion( m_morph, struct_elem );

    matrixtoMat( m_morph, f_morph );

    imwrite( folder_path + true_name + "_morph_erosion.jpg", f_morph );

    //Display
    // displayMat( f_morph, "Erosion Operation" );
}

void ImageViewer:: Dilation(){

    string true_name = sfilename.substr(23);
    true_name = true_name.substr(0, true_name.find(".") );

    Mat f_morph = imread( sfilename, 0);

    threshold( f_morph, 150);

    m_int m_morph( f_morph.rows, vector<int>(f_morph.cols) );

    vector<int> v_struct_elem = { 0, 1, 0,
                                 1, 1, 1,
                                 0, 1, 0};

    m_int struct_elem;

    vectomatrix( v_struct_elem, struct_elem );

    Mattomatrix( f_morph, m_morph );

    morph_dilation( m_morph, struct_elem );

    matrixtoMat( m_morph, f_morph );

    imwrite( folder_path + true_name + "_morph_dilation.jpg", f_morph );

    //Display
    // displayMat( f_morph, "Dilation Operation" );

}

void ImageViewer::createActions()
{
    openAct = new QAction(tr("&Open..."), this);
    openAct->setShortcut(tr("Ctrl+O"));
    connect(openAct, SIGNAL(triggered()), this, SLOT(open()));

    printAct = new QAction(tr("&Print..."), this);
    printAct->setShortcut(tr("Ctrl+P"));
    printAct->setEnabled(false);
    connect(printAct, SIGNAL(triggered()), this, SLOT(print()));

    exitAct = new QAction(tr("E&xit"), this);
    exitAct->setShortcut(tr("Ctrl+Q"));
    connect(exitAct, SIGNAL(triggered()), this, SLOT(close()));

    zoomInAct = new QAction(tr("Zoom &In (25%)"), this);
    zoomInAct->setShortcut(tr("Ctrl++"));
    zoomInAct->setEnabled(false);
    connect(zoomInAct, SIGNAL(triggered()), this, SLOT(zoomIn()));

    zoomOutAct = new QAction(tr("Zoom &Out (25%)"), this);
    zoomOutAct->setShortcut(tr("Ctrl+-"));
    zoomOutAct->setEnabled(false);
    connect(zoomOutAct, SIGNAL(triggered()), this, SLOT(zoomOut()));

    normalSizeAct = new QAction(tr("&Normal Size"), this);
    normalSizeAct->setShortcut(tr("Ctrl+S"));
    normalSizeAct->setEnabled(false);
    connect(normalSizeAct, SIGNAL(triggered()), this, SLOT(normalSize()));

    fitToWindowAct = new QAction(tr("&Fit to Window"), this);
    fitToWindowAct->setEnabled(false);
    fitToWindowAct->setCheckable(true);
    fitToWindowAct->setShortcut(tr("Ctrl+F"));
    connect(fitToWindowAct, SIGNAL(triggered()), this, SLOT(fitToWindow()));

/*
    QAction *Histogram;
    QAction *EqHistogram;
    QAction *Filter1;
    QAction *Filter2;
    QAction *Fourier;
    QAction *LowPass;
    QAction *HighPass;
    QAction *Erosion;
    QAction *Dilation;
*/

    qHistogram = new QAction(tr("&Histogram"), this);
    qEqHistogram = new QAction(tr("&EqHistogram"), this);

    qFilter1 = new QAction(tr("&Mean Filter"), this);
    qFilter2 = new QAction(tr("&2nd Derivative Mask"), this);
    qFilter3 = new QAction(tr("&Median Filter"), this);

    qFourier = new QAction(tr("&Fourier"), this);
    qLowPass = new QAction(tr("&LowPass"), this);
    qHighPass = new QAction(tr("&HighPass"), this);

    qErosion = new QAction(tr("&Erosion"), this);
    qDilation = new QAction(tr("&Dilation"), this);


    qHistogram->setEnabled(false);
    qEqHistogram->setEnabled(false);

    qFilter1->setEnabled(false);
    qFilter2->setEnabled(false);
    qFilter3->setEnabled(false);

    qFourier->setEnabled(false);
    qLowPass->setEnabled(false);
    qHighPass->setEnabled(false);

    qErosion->setEnabled(false);
    qDilation->setEnabled(false);


    connect(qHistogram, SIGNAL(triggered()), this, SLOT(Histogram()));
    connect(qEqHistogram, SIGNAL(triggered()), this, SLOT(EqHistogram()));

    connect(qFilter1, SIGNAL(triggered()), this, SLOT(Filter1()));
    connect(qFilter2, SIGNAL(triggered()), this, SLOT(Filter2()));
    connect(qFilter3, SIGNAL(triggered()), this, SLOT(Filter3()));

    connect(qFourier, SIGNAL(triggered()), this, SLOT(Fourier()));
    connect(qLowPass, SIGNAL(triggered()), this, SLOT(LowPass()));
    connect(qHighPass, SIGNAL(triggered()), this, SLOT(HighPass()));

    connect(qErosion, SIGNAL(triggered()), this, SLOT(Erosion()));
    connect(qDilation, SIGNAL(triggered()), this, SLOT(Dilation()));

    aboutAct = new QAction(tr("&About"), this);
    connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));

    aboutQtAct = new QAction(tr("About &Qt"), this);
    connect(aboutQtAct, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
}

void ImageViewer::createMenus()
{
    fileMenu = new QMenu(tr("&File"), this);
    fileMenu->addAction(openAct);
    fileMenu->addAction(printAct);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAct);

    viewMenu = new QMenu(tr("&View"), this);
    viewMenu->addAction(zoomInAct);
    viewMenu->addAction(zoomOutAct);
    viewMenu->addAction(normalSizeAct);
    viewMenu->addSeparator();
    viewMenu->addAction(fitToWindowAct);

     //New 4 Menus
    fileMenu1 = new QMenu(tr("&Equalization"), this);
    fileMenu1->addAction(qHistogram);
    //fileMenu1->addAction(qEqHistogram);
    fileMenu1->addSeparator();
    fileMenu1->addAction(exitAct);

    fileMenu2 = new QMenu(tr("&Filters"), this);
    fileMenu2->addAction(qFilter1);
    fileMenu2->addAction(qFilter2);
    fileMenu2->addAction(qFilter3);
    fileMenu2->addSeparator();
    fileMenu2->addAction(exitAct);

    fileMenu3 = new QMenu(tr("&Fourier"), this);
    fileMenu3->addAction(qFourier);
    fileMenu3->addAction(qLowPass);
    fileMenu3->addAction(qHighPass);
    fileMenu3->addSeparator();
    fileMenu3->addAction(exitAct);

    fileMenu4 = new QMenu(tr("&Morphologycal"), this);
    fileMenu4->addAction(qErosion);
    fileMenu4->addAction(qDilation);
    fileMenu4->addSeparator();
    fileMenu4->addAction(exitAct);

    helpMenu = new QMenu(tr("&Help"), this);
    helpMenu->addAction(aboutAct);
    helpMenu->addAction(aboutQtAct);

    menuBar()->addMenu(fileMenu);
    menuBar()->addMenu(viewMenu);

    menuBar()->addMenu(fileMenu1);
    menuBar()->addMenu(fileMenu2);
    menuBar()->addMenu(fileMenu3);
    menuBar()->addMenu(fileMenu4);

    menuBar()->addMenu(helpMenu);
}



void ImageViewer::updateActions()
{
    zoomInAct->setEnabled(!fitToWindowAct->isChecked());
    zoomOutAct->setEnabled(!fitToWindowAct->isChecked());
    normalSizeAct->setEnabled(!fitToWindowAct->isChecked());
}



void ImageViewer::scaleImage(double factor)
{
    Q_ASSERT(imageLabel->pixmap());
    scaleFactor *= factor;
    imageLabel->resize(scaleFactor * imageLabel->pixmap()->size());

    adjustScrollBar(scrollArea->horizontalScrollBar(), factor);
    adjustScrollBar(scrollArea->verticalScrollBar(), factor);

    zoomInAct->setEnabled(scaleFactor < 3.0);
    zoomOutAct->setEnabled(scaleFactor > 0.333);
}



void ImageViewer::adjustScrollBar(QScrollBar *scrollBar, double factor)
{
    scrollBar->setValue(int(factor * scrollBar->value()
                            + ((factor - 1) * scrollBar->pageStep()/2)));
}

