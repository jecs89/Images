#include <iostream>
#include <math.h>
#include <random>

#include <iomanip>

#include <unistd.h>   //_getch*/
#include <termios.h>  //_getch*/
#include <vector>

#define tampob 3
#define PI 3.14156

double lim1 = -1;
double lim2 = 1;

#define nrogen 3

#define tipo 2 // max(1), min(2)

using namespace std;

typedef struct particula_   //estructura de particula
{
	vector<double> valor;				
	double aptitud;			 
	vector<double> velocidad;

}particula;

particula mejorglobal ; vector< particula > mejorlocal (tampob);

void inicializar( vector<particula>& actual); 
double aptitud( vector<double> &    valor);               
void calvel( vector<particula> &actual);
double vab( double valor);    


int input_num = 4,
    hidden_num = 20,
    output_num = 4;

#define ihPeso input_num*hidden_num 
#define hoPeso hidden_num*output_num
#define nvar ( ihPeso + hoPeso )

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

template <typename T>
void print( vector<T>& vec);


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

    Delta  = create_array( output_num );
    HDelta = create_array( hidden_num );

    HBias = create_array( hidden_num );
    OBias = create_array( output_num );

    InputNudo = aInputNudo;
    Target = aTarget;

    //Inicialización de vectores según el número de variables
    for (int i = 0; i < tampob; ++i){

        mejorlocal[i].valor.resize(nvar);

        actual[i].valor.resize(nvar);
        actual[i].velocidad.resize(nvar);
    }

    //Inicialización global
    mejorglobal.valor.resize(nvar);
    init_rand( lim1, lim2, mejorglobal.valor );

    int idx = 0;
    for( int x = 0; x < input_num; ++x){
        for( int y = 0; y < hidden_num; ++y){
            InputPeso[x][y] = mejorglobal.valor[idx] ;
            idx++;
        }
    }

    idx = 0;
    for( int x = 0; x < hidden_num; ++x){
       for( int y = 0; y < output_num; ++y){
            HiddenPeso[x][y] = mejorglobal.valor[idx] ;
            idx++;
        }
    }

    FeedForward();

    mejorglobal.aptitud = aptitud( mejorglobal.valor );

    // cout << "vec global: ";
    // print(mejorglobal.valor);
    cout << mejorglobal.aptitud << endl;

   // mejorglobal[0] = mejorglobal[1] = 0;

    cout<<"generacion 0 \n";

    int dec = 15; 

    cout << setw(dec*2) << "VALOR" << setw(dec) << "PAPTITUD" << setw(dec) << "AAPTITUD" << setw(dec*2) << "VELOCIDAD" << setw(dec*2) << "MEJOR LOCAL" << endl;

    for( int i = 0; i < tampob; i++){

        init_rand(lim1, lim2, actual[i].valor);
        init_rand(lim1/100, lim2/100, actual[i].velocidad);
        // actual[i].aptitud = aptitud( actual[i].valor );

        for( int j = 0; j < nvar; ++j){
            mejorlocal[i].valor[j] = actual[i].valor[j];
        }

        int idx = 0;
        for( int x = 0; x < input_num; ++x){
            for( int y = 0; y < hidden_num; ++y){
                InputPeso[x][y] = actual[i].valor[idx] ;
                idx++;
            }
        }

        idx = 0;
        for( int x = 0; x < hidden_num; ++x){
            for( int y = 0; y < output_num; ++y){
                HiddenPeso[x][y] = actual[i].valor[idx] ;
                idx++;
            }
        }

        FeedForward();
        
        actual[i].aptitud = aptitud( actual[i].valor );
        mejorlocal[i].aptitud = aptitud( mejorlocal[i].valor );

        if( tipo == 1){ 
            if( mejorlocal[i].aptitud > mejorglobal.aptitud ) { 
                for( int j = 0; j < nvar; ++j){
                    mejorglobal.valor[j] = mejorlocal[i].valor[j];
                }
                mejorglobal.aptitud = mejorlocal[i].aptitud;
            }
        }

        else if( tipo == 2){ 
            // cout << "Traza\n";
            // cout << actual[i].aptitud << "\t" << mejorglobal.aptitud << endl;
            if( mejorlocal[i].aptitud < mejorglobal.aptitud  ) { 

                for( int j = 0; j < nvar; ++j){
                    mejorglobal.valor[j] = mejorlocal[i].valor[j];
                }
                mejorglobal.aptitud = mejorlocal[i].aptitud;
            }
        }
             
        cout << setw(dec) << actual[i].valor[0] << setw(dec) << actual[i].valor[1] 
             << setw(dec) << actual[i].aptitud << setw(dec) << aptitud(actual[i].valor)
             << setw(dec) << actual[i].velocidad[0] << setw(dec) << actual[i].velocidad[1] ;
                 
        cout << setw(dec) << mejorlocal[i].valor[0] << setw(dec) << mejorlocal[i].valor[1] << "\n";
    }   
    
    cout << "mejor global: " << mejorglobal.valor[0] << "\t" << mejorglobal.valor[1] << "\t aptitud: " << mejorglobal.aptitud << endl;

}

//calculamos los nuevos valores y velocidades
void calvel( vector<particula> &actual){

    default_random_engine rng(random_device{}());       
    uniform_real_distribution<double> dist( lim1, lim2 );

    //tem1 y tem2 valores aleatorios necesarios    
    double tem1 = 0, tem2 = 0; //double signo = tampob/2;

    int dec = 15; 

    //iteramos sobre las n-1 generaciones
    for( int gen = 0; gen < nrogen; gen++) {

        // getch();

       cout<<"Generacion: " << gen + 1 << endl; 

       cout << setw(dec) << "R1" << setw(dec) << "R2" << setw(dec*2) << "VALOR" << setw(dec) << "PAPTITUD" << setw(dec) << "AAPTITUD" << setw(dec*2) << "VELOCIDAD" << setw(dec*2) << "MEJOR LOCAL" << endl;

        // init_ptr( -1, 1, InputPeso, input_num, hidden_num );
        // init_ptr( -1, 1, HiddenPeso, hidden_num, output_num );
        
        //iteramos sobre los tampob de particulas
        for( int i = 0; i < tampob; i++){

            tem1 = dist(rng); 
            tem2 = dist(rng);

            //calculamos velocidad nueva y posicion nueva
            vector<double> temp_vel(nvar);

            for (int j = 0; j < nvar; ++j){
                temp_vel[j] = actual[i].velocidad[j];
            }


            // cout << "Temporal: " << actual[i].velocidad << "\n";

            for (int j = 0; j < nvar; ++j){
                actual[i].velocidad[j] = actual[i].velocidad[j] + tem1 * ( mejorlocal[i].valor[j]- actual[i].valor[j] ) + tem2 * ( mejorglobal.valor[j] - actual[i].valor[j] );
            }
                                  
            vector<double> temp_val(nvar);

            for (int j = 0; j < nvar; ++j){
                temp_val[j] = actual[i].valor[j];
            }

            // cout << "Temporal: " << actual[i].valor << "\n";

            for (int j = 0; j < nvar; ++j){
                actual[i].valor[j] = actual[i].valor[j] + actual[i].velocidad[j];
            }

            // cout << "Después: " << actual[i].valor << "\n";

            bool answer = true, answer2 = true;
            for (int j = 0; j < nvar; ++j){
                if(actual[i].valor[j] < lim1){
                    answer = false;     break;
                }
            }

            for (int j = 0; j < nvar; ++j){
                if(actual[i].valor[j] > lim2){
                    answer2 = false;     break;
                }
            }

            if( !answer || !answer2 ){

                for (int j = 0; j < nvar; ++j){
                    actual[i].valor[j] = temp_val[j];
                    actual[i].velocidad[j] = actual[i].velocidad[j] * -1 ;    
                }
            }

            // actual[i].aptitud = aptitud( actual[i].valor );

            // cout<< setw(dec) << tem1 << setw(dec) << tem2 << setw(dec) << actual[i].valor[0] 
            //     << setw(dec) << actual[i].valor[1] << setw(dec) << (actual[i].aptitud) 
            //     << setw(dec) << aptitud(actual[i].valor)  << setw(dec) << actual[i].velocidad[0] 
            //     << setw(dec) << actual[i].velocidad[1];


            int idx = 0;
            for( int x = 0; x < input_num; ++x){
                for( int y = 0; y < hidden_num; ++y){
                    InputPeso[x][y] = actual[i].valor[idx] ;
                    idx++;
                }
            }

            idx = 0;
            for( int x = 0; x < hidden_num; ++x){
                for( int y = 0; y < output_num; ++y){
                    HiddenPeso[x][y] = actual[i].valor[idx] ;
                    idx++;
                }
            }

            FeedForward();

            actual[i].aptitud = aptitud( actual[i].valor );
            mejorlocal[i].aptitud = aptitud( mejorlocal[i].valor );
            
            // mejorlocal[i].aptitud = aptitud(mejorlocal[i].valor);
            // mejorglobal.aptitud = aptitud(mejorglobal.valor);

            
            //calculamos mejorlocal y mejorglobal
            if( tipo == 1){ 
                cout << "Entreiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii\n";
                
                if( mejorlocal[i].aptitud < actual[i].aptitud ){
                    for( int j = 0; j < nvar; ++j){
                       mejorlocal[i].valor[j] = actual[i].valor[j];
                    }
                    mejorlocal[i].aptitud = actual[i].aptitud;
                }

                if( mejorlocal[i].aptitud > mejorglobal.aptitud ) {
                    for( int j = 0; j < nvar; ++j){
                       mejorglobal.valor[j] = mejorlocal[i].valor[j];
                    }
                    mejorglobal.aptitud = mejorlocal[i].aptitud;
                }
            }

            else if( tipo == 2){ 
                // cout << "//" << aptitud(mejorlocal[i]) << "vs" << aptitud(actual[i].valor) << endl;
                if( mejorlocal[i].aptitud > actual[i].aptitud ){
                    for( int j = 0; j < nvar; ++j){
                       mejorlocal[i].valor[j] = actual[i].valor[j];
                    }
                    mejorglobal.aptitud = actual[i].aptitud;
                }

                // cout << "//" << aptitud(mejorlocal[i]) << "vs" << aptitud(mejorglobal) << endl;

                if( mejorlocal[i].aptitud < mejorglobal.aptitud ) {
                    for( int j = 0; j < nvar; ++j){
                       mejorglobal.valor[j] = mejorlocal[i].valor[j];
                    }
                    mejorglobal.aptitud = mejorlocal[i].aptitud;
                }
            }

            cout<< setw(dec) << tem1 << setw(dec) << tem2 << setw(dec) << actual[i].valor[0] 
                << setw(dec) << actual[i].valor[1] << setw(dec) << (actual[i].aptitud) 
                << setw(dec) << aptitud(actual[i].valor)  << setw(dec) << actual[i].velocidad[0] 
                << setw(dec) << actual[i].velocidad[1];
                    
            cout << setw(dec) << mejorlocal[i].valor[0] << setw(dec) << mejorlocal[i].valor[1] << endl;

        }       

        // mejorglobal.aptitud = aptitud( mejorglobal.valor);
       
       cout << "mejor global generacion "<< gen+1 << " : " << mejorglobal.valor[0] << "\t" << mejorglobal.valor[1] << "\t aptitud: " << mejorglobal.aptitud << endl;

    }
    
        cout << "mejor global: " << mejorglobal.valor[0] << "\t" << mejorglobal.valor[1] << "\t aptitud: " << mejorglobal.aptitud << endl;

    // return mejorglobal;
}

//funcion de aptitud utilizada
double aptitud(vector<double>& valor){
    
    double ans = 0;
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

    ans = Error;

    return ans;
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

template <typename T>
void print(vector<T>& vec){
    for( int i = 0; i < vec.size(); ++i){
        cout << vec[i] << " ";
    }
}