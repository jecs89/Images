#include <iostream>
#include <math.h>
#include <random>

#include <iomanip>

#include <unistd.h>   //_getch*/
#include <termios.h>  //_getch*/
#include <vector>

#define tampob 10
#define PI 3.14156

double lim1 = -1;
double lim2 = 1;

#define nrogen 5

#define tipo 1 // max(1), min(2)

#define nvar 2

using namespace std;

vector<double> mejorglobal ; vector< vector< double > > mejorlocal (tampob);

typedef struct particula_   //estructura de particula
{
	vector<double> valor;				
	double aptitud;			 
	vector<double> velocidad;

}particula;

void inicializar( vector<particula>& actual); 
double aptitud( vector<double> &    valor);               
void calvel( vector<particula> &actual);
double vab( double valor);    


int input_num = 4,
    hidden_num = 20,
    output_num = 4;

double Error;     //error total en la red
double Err;

double** InputPeso,** HiddenPeso; 
double* InputNudo, * HiddenNudo, *OutputNudo, *Target, *Delta, *HDelta, *HBias, *OBias;

double  aInputNudo[]  = { 0, 30, 60, 90 },
        // aHiddenNudo[] = { 0, 0, 0, 0, 0 },
        //aOutputNudo[] = {   },
        aTarget[]     = { 0, 0.5, 0.8660, 1 };

void init_rand(double lim1, double lim2, vector<double>& vec);
char getch();
double *create_array( int Col );
double **create_array( int Row, int Col );
void init_ptr( double lim1, double lim2, double**& ptr, int rows, int cols);
void init_ptr( double lim1, double lim2, double*& ptr, int size);


// void arrtoptr(double arr[], double* &ptr, int size);
// void ptrtoarr(double* &ptr, double(& arr)[], int size);

void FeedForward(void);
double Sigmoid( double num );
double Random( int High, int Low );


int main(int argc, char *argv[])
{
    vector<particula> poblacion(tampob) ;
    inicializar(poblacion);

    // getch(); 

    calvel(poblacion);
    
    return 0;
}

//queremos minimizar entonces menor <
void inicializar( vector<particula>& actual){

    InputPeso = create_array( input_num, hidden_num );
    HiddenPeso = create_array( hidden_num, output_num );

    InputNudo = create_array( input_num );
    HiddenNudo = create_array( hidden_num );
    OutputNudo = create_array( output_num );

    Target = create_array( output_num );
    Delta  = create_array( output_num );
    HDelta = create_array( hidden_num );

    InputNudo = aInputNudo;
    // HiddenNudo = aHiddenNudo;
    //OutputNudo = aOutputNudo;
    Target = aTarget;

    

    //Inicialización de vectores según el número de variables
    for (int i = 0; i < tampob; ++i){
        //mejorglobal[i].resize(nvar);

        mejorlocal[i].resize(nvar);

        actual[i].valor.resize(nvar);
        actual[i].velocidad.resize(nvar);
    }

    //Inicialización global
    mejorglobal.resize(nvar);
    init_rand( lim1, lim2, mejorglobal );
    //mejorglobal[0] = mejorglobal[1] = 0;

    cout<<"generacion 0 \n";

    int dec = 15; 

    cout << setw(dec*2) << "VALOR" << setw(dec) << "APTITUD" << setw(dec*2) << "VELOCIDAD" << setw(dec*2) << "MEJOR LOCAL" << endl;

    for( int i = 0; i < tampob; i++){

        init_rand(lim1, lim2, actual[i].valor);
        actual[i].aptitud = aptitud(actual[i].valor );
        init_rand(lim1/10, lim2/10, actual[i].velocidad);

        FeedForward();
        
        for( int j = 0; j < mejorlocal[0].size(); ++j){
            mejorlocal[i][j] = actual[i].valor[i];
        }
        
        if( tipo == 1){ 
            if( aptitud(actual[i].valor) > aptitud(mejorglobal) ) { 
            
                for( int j = 0; j < mejorlocal[0].size(); ++j){
                    mejorlocal[i][j] = actual[i].valor[i];
                }

            }
        }

        else if( tipo == 2){ 
            if( aptitud(actual[i].valor) < aptitud(mejorglobal)  ) { 
                
                for( int j = 0; j < mejorlocal[0].size(); ++j){
                    mejorlocal[i][j] = actual[i].valor[i];
                }

            }
        }
             
        cout << setw(dec) << actual[i].valor[0] << setw(dec) << actual[i].valor[1] << setw(dec) <<actual[i].aptitud << setw(dec) << actual[i].velocidad[0] << setw(dec) << actual[i].velocidad[1] ;
                 
        // if( mejorlocal[i] > actual[i].valor && mejorlocal[i] > 0 ) {
        //     mejorlocal[i] = actual[i].valor;
        // }
             
        cout << setw(dec) << mejorlocal[i][0] << setw(dec) << mejorlocal[i][1] << "\n";
    }
    
    cout << "mejor global: " << mejorglobal[0] << "\t" << mejorglobal[1] << endl;

    //return actual;
}

//calculamos los nuevos valores y velocidades
void calvel( vector<particula> &actual){

    default_random_engine rng(random_device{}());       
    uniform_real_distribution<double> dist( lim1, lim2 );

    //tem1 y tem2 valores aleatorios necesarios    
    double tem1 = 0, tem2 = 0; //double signo = tampob/2;

    int dec = 15; 

    //iteramos sobre las n-1 generaciones
    for( int j=0; j<nrogen; j++) {

        // getch();

       cout<<"Generacion: "<<j+1<<endl; 

       //cout << setw(dec) << "R1" << setw(dec) << "R2" << setw(dec) << "VALOR" << setw(dec) << "APTITUD" << setw(dec) << "VELOCIDAD" << setw(dec) << "MEJOR LOCAL" << endl;
        
        //iteramos sobre los tampob de particulas
        for( int i=0; i<tampob; i++){

            int ind = i; 
            tem1 = dist(rng); 
            tem2 = dist(rng);
            
            //calculamos velocidad nueva y posicion nueva

            vector<double> temp_vel = actual[i].velocidad;

            // cout << "Temporal: " << actual[i].velocidad << "\n";

            actual[i].velocidad[0] = actual[ind].velocidad[0] + tem1 * ( mejorlocal[i][0]- actual[i].valor[0] ) + tem2 * ( mejorglobal[0] - actual[i].valor[0] );
            actual[i].velocidad[1] = actual[ind].velocidad[1] + tem1 * ( mejorlocal[i][1]- actual[i].valor[1] ) + tem2 * ( mejorglobal[1] - actual[i].valor[1] );
                                  
            vector<double> temp_val = actual[ind].valor;

            // cout << "Temporal: " << actual[i].valor << "\n";

            actual[ind].valor[0] = actual[ind].valor[0] + actual[i].velocidad[0];
            actual[ind].valor[1] = actual[ind].valor[1] + actual[i].velocidad[1];

            // cout << "Después: " << actual[i].valor << "\n";

            if( actual[ind].valor[0] < lim1 || actual[ind].valor[1] < lim1 || actual[ind].valor[0] > lim2 || actual[ind].valor[1] > lim2 ){
                actual[ind].valor = temp_val;
                actual[i].velocidad[0] = actual[i].velocidad[0] * -1 ;
                actual[i].velocidad[1] = actual[i].velocidad[1] * -1 ;
            }

            // cout<< setw(dec) << tem1 << setw(dec) << tem2 << setw(dec) << setw(dec) << actual[i].valor << setw(dec) << aptitud(actual[i].valor)  << setw(dec) << actual[i].velocidad;
            
            //calculamos mejorlocal y mejorglobal
            if( tipo == 1){ 
                
                if( aptitud(mejorlocal[i]) < aptitud(actual[i].valor) ){
                    mejorlocal[i] = actual[i].valor;
                }

                if( aptitud(mejorlocal[i]) > aptitud(mejorglobal) ) {
                    mejorglobal = mejorlocal[i];
                }
            }

            else if( tipo == 2){ 
                
                if( aptitud(mejorlocal[i]) > aptitud(actual[i].valor) ){
                    mejorlocal[i] = actual[i].valor;
                }

                if( aptitud(mejorlocal[i]) < aptitud(mejorglobal) ) {
                    mejorglobal = mejorlocal[i];
                }
            }
                    
            // cout << setw(dec) << mejorlocal[i] << endl;

        }       
       
       cout << "mejor global generacion "<< j << " : " << mejorglobal[0] << "\t" << mejorglobal[1] << endl;

    }
    
    cout << "mejor global: " << mejorglobal[0] << "\t" << mejorglobal[1] << endl;

    // return mejorglobal;
}

//funcion de aptitud utilizada
double aptitud(vector<double>& valor){
    
    //contadores
    int i, j;
    // ajuste del error a 0
    Error = 0;

    //Variable auxiliar para tener el valor del error
    double error_sum = 0.0f;

    //calculo del error a la capa de salida
    for ( i = 0; i < output_num; i++ ) {
        Err = ( Target[i] - OutputNudo[i] );
        Delta[i] = ( 1 - OutputNudo[i] )*OutputNudo[i] * Err;
        Error += 0.5f * Err * Err;
    }

    //calculo del error a la capa oculta
    for ( i = 0; i < hidden_num; i++ ) {
        for ( j = 0; j < output_num; j++ ) {
            error_sum += Delta[j]*HiddenPeso[i][j];
        }
        HDelta[i] = ( 1 - HiddenNudo[i] )*HiddenNudo[i] * error_sum;
        //reinicio de la variable auxiliar
        error_sum = 0.0f;
    }

    return Error;
}

double vab(double valor){
    if( valor < 0 ) valor = valor * -1;
    
    return valor;
}

char getch(){
    /*#include <unistd.h>   //_getch*/
    /*#include <termios.h>  //_getch*/
    char buf=0;
    struct termios old={0};
    fflush(stdout);
    if(tcgetattr(0, &old)<0)
        perror("tcsetattr()");
    old.c_lflag&=~ICANON;
    old.c_lflag&=~ECHO;
    old.c_cc[VMIN]=1;
    old.c_cc[VTIME]=0;
    if(tcsetattr(0, TCSANOW, &old)<0)
        perror("tcsetattr ICANON");
    if(read(0,&buf,1)<0)
        perror("read()");
    old.c_lflag|=ICANON;
    old.c_lflag|=ECHO;
    if(tcsetattr(0, TCSADRAIN, &old)<0)
        perror ("tcsetattr ~ICANON");
    printf("%c\n",buf);
    return buf;
 }


void init_rand(double lim1, double lim2, vector<double>& vec){
    
    default_random_engine rng(random_device{}());       
    uniform_real_distribution<double> dist( lim1, lim2 );

    for( int i = 0; i < vec.size(); ++i ){
        vec[i] = dist(rng);
    }
}

double *create_array( int Col ){
    double *array = new double[Col];
    return array;
}

double **create_array( int Row, int Col ){
    double **array = new double*[Row];
    for ( int i = 0; i < Row; i++ )
        array[i] = new double[Col];
    return array;
}

// void arrtoptr(double arr[], double* &ptr, int size){
//     for( int i = 0 ; i < size ; ++i ){
//         ptr[i] = arr[i];
//     }
// }

// void ptrtoarr(double* &ptr, double(& arr)[], int size){
//     for( int i = 0 ; i < size ; ++i ){
//         arr[i] = ptr[i];
//     }   
// }

void FeedForward(void){

    // variables para los contadores
    int i, j;
    // variable auxiliar para guardar el resultado de la suma de los pesos por las entradas
    double synapse_sum = 0.0f;

    //retroalimenta la capa oculta
    for ( i = 0; i < hidden_num; i++ ) {
        for ( j = 0; j < input_num; j++ ) {
            synapse_sum += InputPeso[j][i]*InputNudo[j];
        }
        HiddenNudo[i] = Sigmoid( synapse_sum + HBias[i]);

        //se reinicia el valor de la variable auxiliar para el siguiente loop
        synapse_sum = 0.0f;
    }

    //retroalimenta la capa de salida
    for ( i = 0; i < output_num; i++ ) {
        for ( j = 0; j < hidden_num; j++ ) {
            synapse_sum += HiddenPeso[j][i]*HiddenNudo[j];
        }
        OutputNudo[i] = Sigmoid( synapse_sum + OBias[i] );

        //reinicio de la variable auxiliar
        synapse_sum = 0.0f;
    }
}

//la funcion de transferencia para la red es la funcion sigmoide
double Sigmoid( double num )
{
    return (double)(1/(1+exp(-num)));
}

//Generador de numeros aleatorios, desde un minimo a un maximo
double Random( int High, int Low )
{
    //se usa la funcion time(NULL) para no tener siempre la misma secuencia de aleatorios
    srand( ( unsigned int )time( NULL ) );
    //retorna el numero aleatorio
    return ( (double)rand()/RAND_MAX) * (High - Low) + Low;
}

void init_ptr( double lim1, double lim2, double**& ptr, int rows, int cols){
    default_random_engine rng(random_device{}());       
    uniform_real_distribution<double> dist( lim1, lim2 );

    for( int i = 0; i < rows; ++i ){
        for (int j = 0; j < cols; ++j){
            ptr[i][j] = dist(rng);
        }
    }

}

void init_ptr( double lim1, double lim2, double*& ptr, int size){
    default_random_engine rng(random_device{}());       
    uniform_real_distribution<double> dist( lim1, lim2 );

    for( int i = 0; i < size; ++i ){
        ptr[i] = dist(rng);
    }
}
