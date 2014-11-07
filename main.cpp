

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>


#include "Functions.h"

// g++ -o main -L /usr/local/cuda-6.5/lib64 `pkg-config opencv --cflags` main.cpp `pkg-config opencv --libs` -std=c++11


int main(int argc, char** argv ){

    DIR *dpdf;
    struct dirent *epdf;

    dpdf = opendir("./");

    vector<Mat> v_image;
    vector<string> v_names;

    int cont = 0;
    
    if (dpdf != NULL){
       while (epdf = readdir(dpdf)){
          string name = epdf->d_name;
          if( isFind(name, postfix)){
            printf("Filename: %s %s",epdf->d_name, "\n");
          
            //Mat reader = imread( name, 1);
            v_names.push_back( name );
            cont ++ ;
          }       
       }
    }

    cout << cont << endl;
    //cout << v_image.size() << endl;

    // for( int i = 0 ; i < v_image.size() ; ++i){
        
    //     string s_tmp = static_cast<ostringstream*>( &(ostringstream() << i) )->str();       

    //     imwrite( s_tmp+".jpg", v_image[i] );
    // }
    

    time_t timer = time(0); 

    for( int i = 0 ; i < cont ; ++i){
        
        function_equalization( v_names[i] );

        function_borders( v_names[i] );
        
        function_sfilters( v_names[i] );
        
        function_tfilters( v_names[i] );
        
        function_morphological( v_names[i] );
        
        function_segmentation( v_names[i] );        

        function_dominantcolor( v_names[i] );

    }
   
    time_t timer2 = time(0); 
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

    waitKey(0);

    return 0;
}
