#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;

int main (int argc, char* argv[])
{
    try
    {
        Mat src_host = cv::imread( argv[1] , CV_LOAD_IMAGE_GRAYSCALE);
        gpu::GpuMat dst, src(src_host);
        //src.upload(src_host);

        gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

        cv::Mat result_host = (Mat)dst;
        imshow("Result", result_host);
        waitKey();
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}