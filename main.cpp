
#include "Functions.h"

int main(int argc, char** argv ){

    time_t timer = time(0); 

    
    function_equalization( argv[1] );

    function_borders( argv[1] );
    
    function_sfilters( argv[1] );
    
    function_tfilters( argv[1] );
    
    function_morphological( argv[1] );
    
    function_segmentation( argv[1] );        

   
    time_t timer2 = time(0); 
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

    waitKey(0);

    return 0;
}
