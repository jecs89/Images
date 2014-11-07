
#include <vector>

using namespace cv;
using namespace std;

typedef vector< vector<double> > matrix;
typedef vector< vector<int> > m_int;

struct my_Complex{
	double real;
	double imag;

	my_Complex( double _r, double _i){
		real = _r ;
		imag = _i ;
	}
};
