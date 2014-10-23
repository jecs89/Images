
#include <vector>

using namespace cv;

typedef vector< vector<double> > matrix;

struct my_Complex{
	double real;
	double imag;

	my_Complex( double _r, double _i){
		real = _r ;
		imag = _i ;
	}
};
