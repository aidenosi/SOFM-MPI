#include <mpi.h>
#include <math.h>
#include <cerrno>
#include <deque>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <stdlib.h>
#include "user-input.h"
#include "CImg/CImg.h"
using namespace std;
using namespace cimg_library;

// Compile: 	mpic++ main.cpp -o sofm_mpi user-input.h -I CImg/ -lX11
// Run:		mpirun -n <p> ./sofm_mpi

int main(){
	struct Pixel{ // Struct to represent pixels
		float red;
		float green;
		float blue;
		int x;
		int y;
		Pixel() : red((double) rand() / (RAND_MAX)), blue((double) rand() / (RAND_MAX)), green((double) rand() / (RAND_MAX)) {}
		Pixel(int red, int green, int blue) : red(red), green(green), blue(blue) {}
		
		float getDistance(Pixel p){
			float diff_red = pow(this->red - p.red, 2);
			float diff_green = pow(this->green - p.green, 2);
			float diff_blue = pow(this->blue - p.blue, 2);
			float distance = sqrt(diff_red + diff_green + diff_blue);
			return distance;
		}
		
		void set_RGB(float new_red, float new_green, float new_blue){
			this->red = new_red;
			this->green = new_green;
			this->blue = new_blue;
		}
		
		void set_coordinates(int x, int y){
			this->x = x;
			this->y = y;
		}
		
		float get_red(){
			return this->red;
		}
		
		float get_green(){
			return this->green;
		}	
		
		float get_blue(){
			return this->blue;
		}
		
		int get_x_coord(){
			return this->x;
		}
		
		int get_y_coord(){
			return this->y;
		}				
		
	};
	
	MPI_Init(NULL, NULL); // Initializes MPI
	int my_rank;
	int p; // Num processes
	int map_width;
	int map_height;
	int num_epochs;
	float neighbourhood_radius;
	float learning_rate;	
	
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Status status;
	MPI_Comm_rank(comm, &my_rank);
	MPI_Comm_size(comm, &p);
	
	// Declaring derived MPI datatype
	MPI_Datatype MPI_Pixel;
	int blocklens[] = {1, 1, 1, 1, 1};
	
	// Getting memory displacement for struct
	MPI_Aint baseaddr, addr1, addr2, addr3, addr4, addr5; 
	MPI_Aint indices[5];
	Pixel addressingPixel = Pixel();
	addressingPixel.set_coordinates(0,0);
	MPI_Get_address(&addressingPixel, &baseaddr);
	MPI_Get_address(&addressingPixel.red, &addr1);
	MPI_Get_address(&addressingPixel.green, &addr2);
	MPI_Get_address(&addressingPixel.blue, &addr3);
	MPI_Get_address(&addressingPixel.x, &addr4);
	MPI_Get_address(&addressingPixel.y, &addr5);
	indices[0] = addr1 - baseaddr;
	indices[1] = addr2 - baseaddr;
	indices[2] = addr3 - baseaddr;
	indices[3] = addr4 - baseaddr;
	indices[4] = addr5 - baseaddr;
	
	MPI_Datatype old_types[] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT, MPI_INT};
	MPI_Type_create_struct(5, blocklens, indices, old_types, &MPI_Pixel);
	MPI_Type_commit(&MPI_Pixel);
	
	// Frequency to save a new image of the map 
	int imgOutputFreq = 20;
	int numImages = 2;

	if (my_rank == 0){
		//Gathering Dimensions of SOFM
		map_width = get_integer_input("Width of SOFM", 50, 2000);
		map_height = get_integer_input("Height of SOFM", 50, 2000);

		// Gathering Learning Parameters for the SOFM
		num_epochs = get_integer_input("Number of Epochs", 1, 100000);
		neighbourhood_radius = get_float_input("Neighbourhood Radius", 1, 150);
		learning_rate = get_float_input("Learning Rate", 0, 1.0);
		
	}
	
	// Broadcast initial values to all processes
	MPI_Bcast(&map_width, 1, MPI_INT, 0, comm);
	MPI_Bcast(&map_height, 1, MPI_INT, 0, comm);
	MPI_Bcast(&num_epochs, 1, MPI_INT, 0, comm);
	MPI_Bcast(&neighbourhood_radius, 1, MPI_FLOAT, 0, comm);
	MPI_Bcast(&learning_rate, 1, MPI_FLOAT, 0, comm);
	
	CImg<unsigned char> map_image(map_width, map_height, 1, 3, 0);  // For image representation of map_arr

	// For timing execution
	double start = MPI_Wtime();
	
	Pixel map_arr[map_width][map_height];
	// Rank 0 will randomly initialize pixels on map_arr whose RGB colour is respresented as a value between 0 and 1.
	if(my_rank == 0){
		for (int i = 0; i < map_width; i++){
			for (int j = 0; j < map_height; j++){
				Pixel new_pixel = Pixel(); // Generate random pixel 
				new_pixel.set_coordinates(i, j);
				map_arr[i][j] = new_pixel;
				const float rand_colour[] = {(int)(new_pixel.red * 255), (int)(new_pixel.green * 255), (int)(new_pixel.blue * 255)};				
				map_image.draw_point(i, j, rand_colour);
			}
		}
		try{
			const char *fn = "images/1.bmp";
			errno = 0;
			map_image.save(fn);
			if(errno != 0){
				cout << "errno: " << errno << endl;
			}
		}
		catch(const std::exception& ex){
			std::cerr << "Error occurred: " << ex.what() << std::endl;
		}
	}
	
	// Begins "Training" on the SOFM over num_epochs
	for(int this_epoch = 1; this_epoch <= num_epochs; this_epoch++){

		// Rank 0 produces new image at regular intervals
		if(my_rank == 0){
			if(this_epoch % imgOutputFreq == 0){ 
				if(this_epoch > 100){ // Output images more often for the first 100 epochs
					imgOutputFreq = 100;
				}
				for(int x = 0; x < map_width; x++){ // Drawing current state of map_arr to image
					for(int y = 0; y < map_height; y++){
						float colour[] = {map_arr[x][y].get_red() * 255, map_arr[x][y].get_green() * 255, map_arr[x][y].get_blue() * 255};
						map_image.draw_point(x, y, colour);
					}
				}
				std::stringstream ss;
				ss << "images/" << numImages << ".bmp";
				std::string filename;
				ss >> filename;
				const char * c = filename.c_str();
				try{
					errno = 0;
					map_image.save(c);
					if(errno != 0){
						cout << "errno: " << errno << endl;
					}
				}
				catch(const std::exception& ex){
					std::cerr << "Error occurred: " << ex.what() << std::endl;
				}
				numImages++;
			}
		}	

		Pixel rand_pixel = Pixel();// Generates randomly initialized pixel to train the SOFM on. 

		// Finds BMU (the pixel in the map_arr that best matches the rand_pixel)
		const float FLOAT_MAX = numeric_limits<float>::max();
		Pixel BMU = Pixel(0, 0, 0);
		float best_distance = FLOAT_MAX;
		int bestX, bestY;
		if(my_rank == 0){ // Rank 0 finds BMU and broadcasts
			for(int i = 0; i < map_width; i++){
				for (int j = 0; j < map_height; j++){
					if(rand_pixel.getDistance(map_arr[i][j]) < best_distance){
						BMU = map_arr[i][j];
						best_distance = rand_pixel.getDistance(map_arr[i][j]);
						BMU.set_coordinates(i, j);
						bestX = i;
						bestY = j;
   					}
				}
			}
			BMU.set_coordinates(bestX, bestY);
		}
		// Broadcast BMU to all processes
		MPI_Bcast(&BMU, 1, MPI_Pixel, 0, comm);
		
		float a = pow(exp(1.0), (-1 * (((this_epoch-1)/(num_epochs)))));

	        // Calculates the Learning Rate for this Epoch
	        float epoch_lr = learning_rate * a;
	
	        // Calculates the Neighbourhood Radius for this Epoch
	        int epoch_nradius = ceil(neighbourhood_radius * a);
	        if (epoch_nradius < 1){
	            epoch_nradius = 1;
	        }
	        
		// Updates the BMU's RGB Values
		float epoch_red = (BMU.red + epoch_lr * (rand_pixel.red - BMU.red));
		float epoch_green = (BMU.green + epoch_lr * (rand_pixel.green - BMU.green));
		float epoch_blue = (BMU.blue + epoch_lr * (rand_pixel.blue - BMU.blue));
		map_arr[BMU.x][BMU.y].set_RGB(epoch_red, epoch_green, epoch_blue);
			
		// Rank 0 updates BMU's' Colour in Image
		if(my_rank == 0){
			float new_colour[] = {(epoch_red * 255), (epoch_green * 255), (epoch_blue * 255)};
			map_image.draw_point(BMU.x, BMU.y, new_colour);	
		}

		// Loops through to get coodinates of neighbouring pixels to be updated.
		deque<int> x_coord_list, y_coord_list;
		if(my_rank == 0){ // Loop iterates top down, middle to left then middle to right
			for (int y = BMU.y - neighbourhood_radius; y < BMU.y + neighbourhood_radius; y++){
				for (int x = BMU.x; (x - BMU.x) * (x - BMU.x) + (y - BMU.y) * (y - BMU.y) <= neighbourhood_radius * neighbourhood_radius; x--){
					if(!((y == BMU.y) && (x == BMU.x)) && (y < map_height) && (y >= 0) && (x < map_width) && (x >= 0)){
						x_coord_list.push_back(map_arr[x][y].get_x_coord());
						y_coord_list.push_back(map_arr[x][y].get_y_coord());
					}
				}
			
				for (int x = BMU.x + 1; (x - BMU.x) * (x - BMU.x) + (y - BMU.y) * (y - BMU.y) <= neighbourhood_radius * neighbourhood_radius; x++){
					if(!((y == BMU.y) && (x == BMU.x)) && (y < map_height) && (y >= 0) && (x < map_width) && (x >= 0)){
						x_coord_list.push_back(map_arr[x][y].get_x_coord());
						y_coord_list.push_back(map_arr[x][y].get_y_coord());					
					}
				}
			}
	    	}
	    	
	    	// Putting all pixels in BMU's neighbourhood into a 1D array
	    	int num_pixels = x_coord_list.size();
	    	MPI_Bcast(&num_pixels, 1, MPI_INT, 0, comm);
    		Pixel pix_arr[num_pixels];
    		if (my_rank == 0){
	    		for(int i = 0; i < num_pixels; i++){
	    			pix_arr[i] = map_arr[x_coord_list.at(i)][y_coord_list.at(i)];
	    		}
    		}
    		
    		// set up for scatterv
    		int *sendcounts = (int *) malloc(sizeof(int)*p);
		int *displacements = (int *) malloc(sizeof(int)*p);
		int displace_index = 0;
    		int count = num_pixels / p;
    		int remainder = num_pixels % p;
    		
    		// Calculating sendcounts and displacements
		for(int i = 0; i < p; i++){
			sendcounts[i] = count;
			if(remainder > 0){
				sendcounts[i]++;
				remainder--;
			}
			displacements[i] = displace_index;
			displace_index += sendcounts[i];
		}

		int sub_size = sendcounts[my_rank];
		Pixel my_pixels[sub_size];
		    	
    		// Scattering pix_arr to processes	
    		MPI_Scatterv(&pix_arr, sendcounts, displacements, MPI_Pixel, &my_pixels, sub_size, MPI_Pixel, 0, comm);
	    	
	    	// Updating RGB values for all pixels in subarray
	    	for(int i = 0; i < sub_size; i++){
	    		float neighbourhood_multiplier = pow(exp(1), -1 * (my_pixels[i].getDistance(BMU) * my_pixels[i].getDistance(BMU)) / (2 * epoch_nradius * epoch_nradius));
			float updated_red = (my_pixels[i].red + neighbourhood_multiplier * epoch_lr * (BMU.red - my_pixels[i].red));
			float updated_green = (my_pixels[i].green + neighbourhood_multiplier * epoch_lr * (BMU.green - my_pixels[i].green));
			float updated_blue = (my_pixels[i].blue + neighbourhood_multiplier * epoch_lr * (BMU.blue - my_pixels[i].blue));
			my_pixels[i].set_RGB(updated_red, updated_green, updated_blue);
	    	}
	    	
	    	// Gathering sub arrays to rank 0
	    	MPI_Gatherv(&my_pixels, sub_size, MPI_Pixel, &pix_arr, sendcounts, displacements, MPI_Pixel, 0, comm);
	    	
	    	free(sendcounts); // Freeing pointers
	    	free(displacements);
		
		if(my_rank == 0){ // Rank 0 updates map_arr
			int c = 0;
			for (int y = BMU.y - neighbourhood_radius; y < BMU.y + neighbourhood_radius; y++){
				for (int x = BMU.x; (x - BMU.x) * (x - BMU.x) + (y - BMU.y) * (y - BMU.y) <= neighbourhood_radius * neighbourhood_radius; x--){
					if(!((y == BMU.y) && (x == BMU.x)) && (y < map_height) && (y >= 0) && (x < map_width) && (x >= 0)){					
						map_arr[x][y] = pix_arr[c];
						c++;
					}
				}
				for (int x = BMU.x + 1; (x - BMU.x) * (x - BMU.x) + (y - BMU.y) * (y - BMU.y) <= neighbourhood_radius * neighbourhood_radius; x++){
					if(!((y == BMU.y) && (x == BMU.x)) && (y < map_height) && (y >= 0) && (x < map_width) && (x >= 0)){
						map_arr[x][y] = pix_arr[c];
						c++;				
					}
				}
			}
	    	}

	}
	
	if(my_rank == 0){ // Output final image and print execution time
		std::stringstream ss;
		ss << "images/" << numImages << ".bmp";
		std::string filename;
		ss >> filename;
		const char * c = filename.c_str();
		try{
			errno = 0;
			map_image.save(c);
			if(errno != 0){
				cout << "errno: " << errno << endl;
			}
		}
		catch(const std::exception& ex){
			std::cerr << "Error occurred: " << ex.what() << std::endl;
		}
		cout<<"Final image saved!"<<endl;
		double time = MPI_Wtime() - start;
		cout<<"Execution time: "<<time<<endl;
		
	}
	MPI_Finalize();
	return 0;
}
