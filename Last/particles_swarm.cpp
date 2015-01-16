#include <iostream>
#include <math.h>
#include <random>

#include <iomanip>

#include <unistd.h>   //_getch*/
#include <termios.h>  //_getch*/


#define tampob 100

#define lim1 -100
#define lim2 100

#define nrogen 100

#define tipo 1 // max(1), min(2)

double mejorglobal ; double *mejorlocal=new double [tampob];

typedef struct particula_   //estructura de particula
{
	double valor;				
	double aptitud;			 
	double velocidad;
	double posicion;
}particula;

using namespace std;

particula* inicializar( particula *actual); 
double aptitud(double valor);               
double calvel( particula* &actual);      
double reducir(double valor);           
double vab(double valor);               

char getch();

int main(int argc, char *argv[])
{
    particula* poblacion = new particula [tampob];
    poblacion = inicializar(poblacion);

    // getch(); 

    int global = calvel(poblacion);
    
    return 0;
}

//queremos minimizar entonces menor <
particula* inicializar( particula *actual){

    default_random_engine rng(random_device{}());       
    uniform_real_distribution<double> dist( lim1, lim2 );

    default_random_engine rng2(random_device{}());       
    uniform_real_distribution<double> dist2( -1 * lim2, lim1 );
    
    double mejor = 0;
    cout<<"generacion 0 \n";

    int dec = 15; 

    cout << setw(dec) << "VALOR" << setw(dec) << "APTITUD" << setw(dec) << "VELOCIDAD" << setw(dec) << "MEJOR LOCAL" << endl;

    mejorglobal = dist(rng);

    for( int i = 0; i < tampob; i++){

        actual[i].valor     = dist(rng);
        actual[i].aptitud   = aptitud(actual[i].valor);
        actual[i].velocidad = dist(rng) / 10 ;
         
        mejorlocal[i] = actual[i].valor;
        
        if( tipo == 1){ 
            if( aptitud(actual[i].valor) > aptitud(mejorglobal) && mejorglobal >= lim1 ) { 
                mejorglobal = actual[i].valor; 
            }
        }

        else if( tipo == 2){ 
            if( aptitud(actual[i].valor) < aptitud(mejorglobal) && mejorglobal >= lim1  ) { 
                mejorglobal = actual[i].valor; 
            }
        }
             
        cout << setw(dec) << actual[i].valor << setw(dec) <<actual[i].aptitud << setw(dec) << actual[i].velocidad;
                 
        // if( mejorlocal[i] > actual[i].valor && mejorlocal[i] > 0 ) {
        //     mejorlocal[i] = actual[i].valor;
        // }
             
        cout << setw(dec) << mejorlocal[i] << "\n";
    }
    
    cout << "mejor global: " << mejorglobal << endl;

    return actual;

}

//calculamos los nuevos valores y velocidades
double calvel( particula* &actual){

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
       
       cout << "mejor global generacion "<< j << " : " << mejorglobal << endl;

    }
    
    cout << "mejor global : " << mejorglobal << endl;

    return mejorglobal;
}

//funcion de aptitud utilizada
double aptitud(double valor){
    return ( 1.0 / ( 1.0 + exp(-valor) ) );
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