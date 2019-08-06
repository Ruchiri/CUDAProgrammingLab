#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_profiler_api.h>
#include <tuple>
#include <iostream>
#include <string.h>

double time_host = 0;
double time_device = 0;

int sample_rounds = 10;

void meanFilter_host(unsigned char* image_matrix,unsigned char* filtered_image_data,int image_width, int image_height, int window_size)
{
    int half_window = (window_size-window_size % 2)/2;
	
    for(int i = 0; i < image_height; i += 1){
        for(int j = 0; j < image_width; j += 1){
            int k = 3*(i*image_height+j);
            int top_boundary;
			int bottom_boundary;
			int left_boundary;
			int right_boundary; 
            if(i-half_window >= 0){
				top_boundary = i-half_window;
			}else{
				top_boundary = 0;
			}
            if(i+half_window <= image_height-1){
				bottom_boundary = i+half_window;
			}else{
				bottom_boundary = image_height-1;
			}
            if(j-half_window >= 0){
				left_boundary = j-half_window;
			}else{
				left_boundary = 0;
			}
            if(j+half_window <= image_width-1){
				right_boundary = j+half_window;
			}else{
				right_boundary = image_width-1;
			}
            double byte1 = 0; 
            double byte2 = 0; 
            double byte3 = 0; 
            
            for(int x = top_boundary; x <= bottom_boundary; x++){
                for(int y = left_boundary; y <= right_boundary; y++){
                    int pos = 3*(x*image_height + y); 
                    byte1 += image_matrix[pos];
                    byte2 += image_matrix[pos+1];
                    byte3 += image_matrix[pos+2];
                }
            }
            int effective_window_size = (bottom_boundary-top_boundary+1)*(right_boundary-left_boundary+1);
            filtered_image_data[k] = byte1/effective_window_size;
            filtered_image_data[k+1] = byte2/effective_window_size;
            filtered_image_data[k+2] = byte3/effective_window_size;

            
        }
    }
   
}

__global__ void meanFilter_device(unsigned char* image_matrix, unsigned char* filtered_image_data, int image_width, int image_height, int window_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
	int half_window = (window_size-window_size % 2)/2;
	
    if (i < image_height && j < image_width){
        int k = 3*(i*image_height+j);
        int top_boundary;
		int bottom_boundary;
		int left_boundary;
		int right_boundary; 
        if(i-half_window >= 0){
			top_boundary = i-half_window;
		}else{
			top_boundary = 0;
		}
        if(i+half_window <= image_height-1){
			bottom_boundary = i+half_window;
		}else{
			bottom_boundary = image_height-1;
		}
        if(j-half_window >= 0){
			left_boundary = j-half_window;
		}else{
			left_boundary = 0;
		}
        if(j+half_window <= image_width-1){
			right_boundary = j+half_window;
		}else{
			right_boundary = image_width-1;
		}
        double byte1 = 0; 
        double byte2 = 0; 
        double byte3 = 0; 
       
        for(int x = top_boundary; x <= bottom_boundary; x++){
            for(int y = left_boundary; y <= right_boundary; y++){
                int pos = 3*(x*image_height + y); 
                byte1 += image_matrix[pos];
                byte2 += image_matrix[pos+1];
                byte3 += image_matrix[pos+2];
            }
        }
        int effective_window_size = (bottom_boundary-top_boundary+1)*(right_boundary-left_boundary+1);
        filtered_image_data[k] = byte1/effective_window_size;
        filtered_image_data[k+1] = byte2/effective_window_size;
        filtered_image_data[k+2] = byte3/effective_window_size;
    }
}


int main(int argc,char **argv)
{
   
    FILE* f = fopen(argv[1], "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); 

    int width, height;
    memcpy(&width, info + 18, sizeof(int));
    memcpy(&height, info + 22, sizeof(int));

    int window_size = strtol(argv[2],NULL,10);
    printf("     Window size: %d\n",window_size);
    printf("Image dimensions: (%d, %d)\n",width,height);
        
    int size = 3 * width * abs(height);
    unsigned char* data = new unsigned char[size]; 
    unsigned char* result_image_data_d;
    unsigned char* result_image_data_h = new unsigned char[size];
    unsigned char* result_image_data_h1 = new unsigned char[size];

    unsigned char* image_data_d;

    fread(data, sizeof(unsigned char), size, f); 
    fclose(f);
   
    int block_size = 32;
    int grid_size = width/block_size;
    dim3 dimBlock(block_size, block_size, 1);
    dim3 dimGrid(grid_size, grid_size, 1);

    
    for(int i = 0; i < sample_rounds; i += 1)
    {
        cudaMalloc((void **)&image_data_d,size*sizeof(unsigned char));
        cudaMalloc((void **)&result_image_data_d,size*sizeof(unsigned char));
        cudaMemcpy(image_data_d,data,size*sizeof(unsigned char),cudaMemcpyHostToDevice);
        
       
        clock_t start_d=clock();
        meanFilter_device <<< dimGrid, dimBlock >>> (image_data_d, result_image_data_d, width, height, window_size);
        cudaThreadSynchronize();

        cudaError_t error = cudaGetLastError();
        if(error!=cudaSuccess)
        {
            fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
            exit(-1);
        }
        clock_t end_d = clock();

        clock_t start_h = clock();
        meanFilter_host(data, result_image_data_h1, width, height, window_size);
        clock_t end_h = clock();

        cudaMemcpy(result_image_data_h,result_image_data_d,size*sizeof(unsigned char),cudaMemcpyDeviceToHost);

        time_host += (double)(end_h-start_h)/CLOCKS_PER_SEC;
        time_device += (double)(end_d-start_d)/CLOCKS_PER_SEC;

        cudaFree(image_data_d);
        cudaFree(result_image_data_d);
    }

    printf("    GPU Time: %f\n",(time_device/sample_rounds));
    printf("    CPU Time: %f\n",(time_host/sample_rounds));
    printf("CPU/GPU time: %f\n",(time_host/time_device));

    
    return 0;
}

