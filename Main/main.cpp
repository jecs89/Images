

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>


#include "Functions.h"

ofstream name_images( folder_path + "name_images.data"); 

// g++ -o main -L /usr/local/cuda-6.5/lib64 `pkg-config opencv --cflags` main.cpp `pkg-config opencv --libs` -std=c++11

bool sorter(const string& left, const string& right)
{
    //go through each column
    for(int i=0; i<left.length() && i<right.length(); ++i ) {
        // if left is "more" return that we go higher
        if( left[i] < right[i])
            return true;
        // if left is "less" return that we go lower
        else if (left[i] > right[i])
            return false;
    }
    // if left is longer, it goes higher
    if (left.size() < right.size())
        return true;
    else //otherwise, left go lower
        return false;
 }

int main(int argc, char** argv ){

    //features.precision(spc_file-1);

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
            //printf("Filename: %s %s",epdf->d_name, "\n");
          
            //Mat reader = imread( name, 1);
            v_names.push_back( name );
            cont ++ ;
          }       
       }
    }

    cout << cont << endl;

    sort( v_names.begin(), v_names.end(), sorter);

    //cout << v_image.size() << endl;

    // for( int i = 0 ; i < v_image.size() ; ++i){
        
    //     string s_tmp = static_cast<ostringstream*>( &(ostringstream() << i) )->str();       

    //     imwrite( s_tmp+".jpg", v_image[i] );
    // }
    

    features << cont << endl;

    time_t timer = time(0); 

    for( int i = 0 ; i < cont ; ++i){

        vt_borders.clear();
        vm_borders.clear();    

        //cout << v_names[i] << endl;
        //cout << ".";
        name_images << v_names[i] << endl;
        
        function_equalization( v_names[i] );

        function_borders( v_names[i] );
        
        //function_sfilters( v_names[i] );
        
        //function_tfilters( v_names[i] );
        
        function_morphological( v_names[i] );
        
        //function_segmentation( v_names[i] );        

        function_dominantcolor( v_names[i] );

        comp_borders();
        
    }

    time_t timer2 = time(0); 
    cout <<"Tiempo total: " << difftime(timer2, timer) << endl;

    waitKey(0);

    return 0;
}
