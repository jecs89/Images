#include <complex>
#include <iostream>
#include <valarray>
#include <vector>
#include <iomanip>
 
const double PI = 3.141592653589793238460;
 
typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

#define space 20

using namespace std;
 
// Cooley–Tukey FFT (in-place)
void fft(CArray& x)
{
    const size_t N = x.size();
    if (N <= 1) return;
 
    // divide
    CArray even = x[std::slice(0, N/2, 2)];
    CArray  odd = x[std::slice(1, N/2, 2)];
 
    // conquer
    fft(even);
    fft(odd);
 
    // combine
    for (size_t k = 0; k < N/2; ++k)
    {
    	double theta = 2 * PI * k / N;
    	Complex th( cos(theta), sin(theta) );

        //Complex t = Complex( cos(theta), sin(theta) ) * odd[k];        

        Complex t( th.real()*odd[k].real() - th.imag() * odd[k].imag() , th.real() *odd[k].imag() + th.imag()*odd[k].real());

        x[k    ] = even[k] + t;
        x[k+N/2] = even[k] - t;
    }
}

// inverse fft (in-place)
void ifft(CArray& x)
{
    // conjugate the complex numbers
    x = x.apply(std::conj);
 
    // forward fft
    fft( x );
 
    // conjugate the complex numbers again
    x = x.apply(std::conj);
 
    // scale the numbers
    x /= x.size();
}

struct my_Complex{
	double real;
	double imag;

	my_Complex( double _r, double _i){
		real = _r ;
		imag = _i ;
	}
};

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

		source[i] 		= my_Complex( even[i].real + t.real, even[i].imag + t.imag ) ;

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

int main()
{
    const Complex test[] = { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
    CArray data(test, 8);
 
    // forward fft
    fft(data);
 
    //cout << "fft" << std::endl;
    /*for (int i = 0; i < 8; ++i){
        cout << data[i] << std::endl;
    }
 */
    // inverse fft
    ifft(data);
 
    cout << std::endl << "ifft" << std::endl;
    for (int i = 0; i < 8; ++i){
        cout << data[i] << std::endl;
    }

    vector<my_Complex> v_complex;

    v_complex.push_back( my_Complex(1.0,0.0) );
    v_complex.push_back( my_Complex(1.0,0.0) );
    v_complex.push_back( my_Complex(1.0,0.0) );
    v_complex.push_back( my_Complex(1.0,0.0) );

    v_complex.push_back( my_Complex(0.0,0.0) );
    v_complex.push_back( my_Complex(0.0,0.0) );
    v_complex.push_back( my_Complex(0.0,0.0) );
    v_complex.push_back( my_Complex(0.0,0.0) );
/*
    for( int i = 0 ; i < v_complex.size(); ++i){
    	cout << v_complex[i].real << " " << v_complex[i].imag << endl;
    }
*/
    fft1d( v_complex );
/*
    for( int i = 0; i < v_complex.size(); ++i){
    	cout << v_complex[i].real << "\t" << v_complex[i].imag << endl;
    }
*/
    ffti1d( v_complex );

    for( int i = 0; i < v_complex.size(); ++i){
    	cout << v_complex[i].real << "\t" << v_complex[i].imag << endl;
    }

    /*complex<double> c1 ( -1, -2);
    complex<double> c2 ( -2, -4);

    cout << c1 * c2 << endl;

    cout << c1.real()*c2.real() - c1.imag() * c2.imag() << "\t" << c1.real() * c2.imag() + c1.imag()*c2.real();
	*/

    
/*
    size_t N = data.size();
    CArray even = data[slice(0, N/2, 2)];
    CArray  odd = data[slice(1, N/2, 2)];


    vector<my_Complex> m_even;
	vector<my_Complex> m_odd;	

	get_values( v_complex, m_even, 0, N/2, 2 );
	get_values( v_complex, m_odd, 1, N/2, 2 );

	cout << " ///////// " << endl;

    for (int i = 0; i < 4; ++i){
        cout << setw( space ) << even[i] << "\t";
        cout << m_even[i].real <<" " << m_even[i].imag << endl;
    }

    cout << " ///////// " << endl;

    for (int i = 0; i < 4; ++i){
        cout  << setw( space ) << odd[i] << "\t";
        cout << m_odd[i].real <<" " << m_odd[i].imag << endl;
    }

	cout << "My first Make \n";
*/
    return 0;
}
