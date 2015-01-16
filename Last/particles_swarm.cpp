#include <iostream>
#include <math.h>
#include <random>

#include <iomanip>

#include <unistd.h>   //_getch*/
#include <termios.h>  //_getch*/
#include <vector>

#define tampob 100

double lim1 = -100;
double lim2 = 100;

#define nrogen 100

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
double aptitud( vector<double> valor);               
void calvel( vector<particula> &actual);
double vab( double valor);    

void init_rand(double lim1, double lim2, vector<double>& vec){
    
    default_random_engine rng(random_device{}());       
    uniform_real_distribution<double> dist( lim1, lim2 );

    for( int i = 0; i < vec.size(); ++i ){
        vec[i] = dist(rng);
    }
}

char getch();

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

    cout<<"generacion 0 \n";

    int dec = 15; 

    cout << setw(dec) << "VALOR" << setw(dec) << "APTITUD" << setw(dec) << "VELOCIDAD" << setw(dec) << "MEJOR LOCAL" << endl;

    for( int i = 0; i < tampob; i++){

        init_rand(lim1, lim2, actual[i].valor);
        actual[i].aptitud = aptitud(actual[i].valor );
        init_rand(lim1/10, lim2/10, actual[i].valor);
        
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
    
    cout << "mejor global: " << mejorglobal[0] << mejorglobal[1] << endl;

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

            double temp_vel = actual[i].velocidad;

            // cout << "Temporal: " << actual[i].velocidad << "\n";

            actual[i].velocidad = actual[ind].velocidad + tem1 * ( mejorlocal[i]- actual[i].valor ) + tem2 * ( mejorglobal - actual[i].valor );
                                  
            double temp_val = actual[ind].valor;

            // cout << "Temporal: " << actual[i].valor << "\n";

            actual[ind].valor = actual[ind].valor + actual[i].velocidad;

            // cout << "Después: " << actual[i].valor << "\n";

            if( actual[ind].valor < lim1 || actual[ind].valor > lim2 ){
                actual[ind].valor = temp_val;
                actual[i].velocidad = actual[i].velocidad * -1 ;
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
       
       cout << "mejor global generacion "<< j << " : " << mejorglobal[0] << mejorglobal[1] << endl;

    }
    
    cout << "mejor global: " << mejorglobal[0] << mejorglobal[1] << endl;

    // return mejorglobal;
}

//funcion de aptitud utilizada
double aptitud(vector<double>& valor){
    double val;

    for (int i = 0; i < valor.size(); ++i)    {
        val += ( 1.0 / ( 1.0 + exp(-valor[i]) ) ) ;
    }

    return val;

    //return sin(valor);
    //return 1/ powf( valor, 2 );
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