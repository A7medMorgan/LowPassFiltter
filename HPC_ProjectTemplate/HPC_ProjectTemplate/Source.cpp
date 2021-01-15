#include <iostream>
#include <math.h>
#include <stdlib.h>
#include<string.h>
#include<msclr\marshal_cppstd.h>
#include <ctime>// include this header 
#include<mpi.h>
#pragma once

#using <mscorlib.dll>
#using <System.dll>
#using <System.Drawing.dll>
#using <System.Windows.Forms.dll>
using namespace std;
using namespace msclr::interop;

#define mcw MPI_COMM_WORLD

void print(int* img, int start_i, int start_j, int end_row, int end_col, int size_col);
void print(float* img, int start_i, int start_j, int end_row, int end_col, int size_col);
int portion_ratio(int num_row, int size);
void specify_index(int row, int* n_row_created, int* start_row_index, int div_ratio, int rank, int size, int kernel_len);
int* border_replication(int* img, int img_row, int img_col, int kernel_len);
int* padding_2row_2col(int* img, int row, int col);
int* portion_image(int* padded_img, int row, int col_extended, int* n_row_created, int div_ratio, int rank, int size, int kernel_len);
int* Matrix_mul(int* img, int img_row, int img_col, float* kernel, int kernal_len);  // kernel is float



int* inputImage(int* w, int* h, System::String^ imagePath) //put the size of image in w & h
{
	int* input;

	int OriginalImageWidth, OriginalImageHeight;

	//*********************************************************Read Image and save it to local arrayss*************************	
	//Read Image and save it to local arrayss

	System::Drawing::Bitmap BM(imagePath);

	OriginalImageWidth = BM.Width;
	OriginalImageHeight = BM.Height;
	*w = BM.Width;
	*h = BM.Height;
	int *Red = new int[BM.Height * BM.Width];
	int *Green = new int[BM.Height * BM.Width];
	int *Blue = new int[BM.Height * BM.Width];
	input = new int[BM.Height*BM.Width];
	for (int i = 0; i < BM.Height; i++)
	{
		for (int j = 0; j < BM.Width; j++)
		{
			System::Drawing::Color c = BM.GetPixel(j, i);

			Red[i * BM.Width + j] = c.R;
			Blue[i * BM.Width + j] = c.B;
			Green[i * BM.Width + j] = c.G;

			input[i*BM.Width + j] = ((c.R + c.B + c.G) / 3); //gray scale value equals the average of RGB values

		}

	}
	return input;
}


void createImage(int* image, int width, int height, int index)
{
	System::Drawing::Bitmap MyNewImage(width, height);

	for (int i = 0; i < MyNewImage.Height; i++)
	{
		for (int j = 0; j < MyNewImage.Width; j++)
		{
			//i * OriginalImageWidth + j
			if (image[i*width + j] < 0)
			{
				image[i*width + j] = 0;
			}
			if (image[i*width + j] > 255)
			{
				image[i*width + j] = 255;
			}
			System::Drawing::Color c = System::Drawing::Color::FromArgb(image[i*MyNewImage.Width + j], image[i*MyNewImage.Width + j], image[i*MyNewImage.Width + j]);
			MyNewImage.SetPixel(j, i, c);
		}
	}
	MyNewImage.Save("..//Data//Output//outputRes" + index + ".png");
	cout << endl;
	cout << "filtered Image Saved " << "as (OutputRes"<<index <<".png)"<< endl;
	cout << endl;
}



int main(int args, char** argv)
{
	// Kernels (3x3)  | (5x5)

	 const int kernel_len_3 = 3;
	float kernel_3[kernel_len_3 * kernel_len_3] = {
		1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,
		1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,
		1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f
	}; 

	const int kernel_len_5 = 5;
	float kernel_5[kernel_len_5 * kernel_len_5] = {
	1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,
	1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,
	1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,
	1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,
	1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f,1.0f / 9.0f
	};

	// choosen kernel
	int kernel_len =0;
	float* kernel = new float{};


	#pragma region parameters

	// global variable
	int col_extend = 0;
	

	int* sub_image = new int{};
	int sub_img_r = 0;

	// for every thread has its own result
	int* matrix_mul;

	//  Master Variable
	const int Master = 0;
	int row = 0, col = 0; // row , col of the original image
	int* filtered_img ;  // the final img
	int start_s = 0, stop_s = 0, TotalTime = 0,start_p=0 ,stop_p=0, time_of_parallel=0;
	int div_ratio = 0;

#pragma endregion


	MPI_Init(NULL, NULL);

	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int size = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//  choose the Kernel size by USER INPUT
#pragma region choose the kernel

	if (rank == Master)
	{
		cout << "choose the size of the kernel (kernel data is(1/9))" << endl;
		cout << "Enter 1 for (3x3) || Enter 2 for (5x5)" << endl;
		int i = 1;
		cin >> i;
		if (i == 1)
		{
			kernel = kernel_3;
			kernel_len = kernel_len_3;
		}
		else {
			kernel = kernel_5;
			kernel_len = kernel_len_5;
		}
	}
	MPI_Bcast(&kernel_len,1,MPI_INT,Master,mcw);
	
	if (rank != Master)
	{
		if (kernel_len == kernel_len_3)
		{
			kernel = kernel_3;
		}
		else if (kernel_len == kernel_len_5 )
		{
			kernel = kernel_5;
		}

	}
#pragma endregion

	// get the image and splited and send to each node
	if (rank == Master)
	{
		std::string img_name = "test";  // The image name

		start_s = clock(); // start the time of the program

		// get the image
		System::String^ imagePath;
		std::string img;
		img = "..//Data//Input//" + img_name + ".png";
		imagePath = marshal_as<System::String^>(img);
		cout << "Image Name : " <<img_name<<".png"<< endl;

		cout << "Number of nodes = (" << size << ")" << endl;

		int ImageWidth = 0, ImageHeight = 0;
		int* imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);

		// determine which is the col ,row

		row = ImageWidth;
		col = ImageHeight;

		//row = ImageHeight;
		//col = ImageWidth;
		cout << "---------printing the kernel---------" <<" kernel length = ("<<kernel_len<<"x"<<kernel_len<<")"<< endl;
		print(kernel, 0, 0, kernel_len, kernel_len, kernel_len);
		cout << endl;

		
		col_extend = col + (kernel_len - 1); // add the num of col to padding the original array


			start_p = clock();  // start the timer of aplly the kerel
			cout << "start the clock & communicatio between the nodes and the master node (Rank = " << Master<<")"<< endl;
		// padding the image
		int* pading_image = border_replication(imageData, row, col,kernel_len);


		// divition ratio calcution
				// calculate the ratio of dividing (rows) of the matrix on every thread equly
		div_ratio = portion_ratio(row,size);

		
#pragma region spliting & sending the data   // split the (padded)image across the clients



	 // send the data by refrence the index of the padded image
			MPI_Request request;
		for (int _rank = 0; _rank < size; _rank++)
		{

			int start_row_index = 0;
			int row_created = 0;
			specify_index(row, &row_created, &start_row_index, div_ratio, _rank, size, kernel_len); // get the starting index of each thread and the num of row it gets

			//cout << "rank:  " << _rank << "  start:    " << start_row_index << "    row_created:" << row_created << endl;

			if (_rank == Master) {

				sub_img_r = row_created;
				sub_image = new int[sub_img_r * col_extend];

				for (int i = start_row_index; i < sub_img_r; i++)
				{
					for (int j = 0; j < col_extend; j++)
					{
						sub_image[(i * col_extend) + j] = pading_image[(i * col_extend) + j];
					}
				}
				continue;
			}

				MPI_Send(&row_created, 1, MPI_INT, _rank, Master, mcw); // send the num of that each thread took
				
				// send the data

				//MPI_Send(&pading_image[(start_row_index * col_extend)], row_created * col_extend, MPI_INT, _rank,(Master + 1), mcw);


				// do not wait for send complete how ever use recv to wait for data accurecy and ensures the correction
				MPI_Isend(&pading_image[(start_row_index * col_extend)], row_created * col_extend, MPI_INT, _rank, (Master + 1), mcw, &request);
		}
		


		/*
		  //send the data by slpit the padded image into sub image (arrays)
		
		MPI_Request request;
		for (int _rank = 0; _rank < size; _rank++)
		{
			int start_index = 0;
			int row_created = 0;

			int* _image= portion_image(pading_image,row,col_extend,&row_created,div_ratio,_rank,size,kernel_len);

			 if (_rank == Master)
			 {
				 sub_img_r = row_created;
				 sub_image = _image;

				 continue;
			 }

			 	MPI_Send(&row_created, 1, MPI_INT, _rank, Master, mcw);
					//MPI_Send(&_image[0], row_created * col_extend, MPI_INT, _rank,Master + 1, mcw);

				MPI_Isend(&_image[0], row_created * col_extend, MPI_INT, _rank, (Master + 1), mcw, &request);
		}*/
#pragma endregion

		free(imageData);
		cout << "----------Start the parallel calculation----------" << endl;
	}

	// global variable
	MPI_Bcast(&col_extend, 1, MPI_INT, Master, mcw);

#pragma region parallel code

	// parallel code start

	if (rank != Master) // recv the data form the master
	{
		MPI_Status stats;
		MPI_Recv(&sub_img_r, 1, MPI_INT, Master, Master, mcw, &stats);

		sub_image = new int[sub_img_r * col_extend];
		MPI_Recv(&sub_image[0], sub_img_r * col_extend, MPI_INT, Master,(Master + 1), mcw, &stats); // wait for send to complete

	}
	

	// start the maltiplication of the matrices
	matrix_mul = Matrix_mul(sub_image, sub_img_r, col_extend, kernel, kernel_len);
	
	free(sub_image);

	// send back the results to the master
	if (rank != Master) {
		int row_rank = (sub_img_r - (kernel_len - 1)); // send back only the needed data and discard the padded row&col

		MPI_Send(&row_rank, 1, MPI_INT, Master, Master, mcw);
		MPI_Send(&matrix_mul[0], row_rank * (col_extend - (kernel_len - 1)), MPI_INT, Master, (Master + 1), mcw);

		free(matrix_mul);
	}

	// parallel code End
#pragma endregion


	// recv back the result  and create the filtered image
	if (rank == Master)
	{
#pragma region recv the calculated data and the data of the master into one array(filtered_img)

		cout << "----------End of the parallel calculation---------- " << endl;
		filtered_img = new int[row * col];
		int row_rank = 0;
		MPI_Status status;

		int start_row_index = 0; // index of the start recv buffer for each rank

		for (int _rank = 0; _rank < size; _rank++)
		{
			if (_rank == Master)
			{
				row_rank = (sub_img_r - (kernel_len - 1));  // nu row of master


				for (int i = start_row_index; i < row_rank; i++)
				{
					for (int j = 0; j < col; j++)
					{
						filtered_img[(i * col) + j] = matrix_mul[(i * col) + j]; // fill the part that master thread calc
					}
				}
				start_row_index += row_rank;

				free(matrix_mul);

				continue;
			}

			MPI_Recv(&row_rank, 1, MPI_INT, _rank, Master, mcw, &status);

			MPI_Recv(&filtered_img[(start_row_index * col)], row_rank * col, MPI_INT, _rank, (Master + 1), mcw, &status);

			start_row_index += row_rank;

		}
		cout << " (successfuly)collecting back the result from all threads" << endl;

#pragma endregion

#pragma region create the filtered image
			stop_p = clock(); // stop the time of aplly the kerel
			cout << "Stop the clock & start creating the filtered image" << endl;

			// create the image

			/*cout << "the image will be save as OutputRes(n) ::" << "Enter number replace (n) :" << "\t";
			int n = 0;
			cin >> n;
		createImage(filtered_img, row, col, n);*/
			createImage(filtered_img, row, col, kernel_len);

			stop_s = clock(); // stop the time of the program

			time_of_parallel = (stop_p - start_p) / float(CLOCKS_PER_SEC) * 1000;
			cout << "Time of communication and parallel calculation :=  " << time_of_parallel << "  ms" << endl;


			TotalTime = (stop_s - start_s) / float(CLOCKS_PER_SEC) * 1000;
			cout << "Time of the entire program (Read & Write the image) :=  " << TotalTime << "  ms" << endl;

			delete[]filtered_img;
#pragma endregion
	}

	MPI_Finalize();
}

		// Function Declaretion


void print(int* img,int start_i,int start_j, int end_row, int end_col,int size_col)
{
	if (img == NULL) return;

	cout << "rows = " << end_row << "  columns = " << end_col << endl;
	for (int i = start_i; i < end_row; i++)
	{
		for (int j = start_j; j < end_col; j++)
		{
			// 2d to 1d array index[(i*col) + j]
			cout << img[(i * size_col) + j] << "\t";
		}
		cout << endl;
	}
}
void print(float* img, int start_i, int start_j, int end_row, int end_col, int size_col)
{
	if (img == NULL) return;

	cout << "rows = " << end_row << "  columns = " << end_col << endl;
	for (int i = start_i; i < end_row; i++)
	{
		for (int j = start_j; j < end_col; j++)
		{
			// 2d to 1d array index[(i*col) + j]
			cout << img[(i * size_col) + j] << "\t";
		}
		cout << endl;
	}
}

int portion_ratio(int num_row, int size)
{
	int temp = 0;
	// divition ratio calcution
	float row_calc_div_f = (float)num_row / (float)size;
	if (fmod(row_calc_div_f, 1) >= 0.5)  // fmod return (num of row that make fraction with the divition) 
	{
		row_calc_div_f += 1;
	}
	temp = (int)row_calc_div_f;
	cout << "\n division_ratio for each node  = " << temp <<"  row"<< endl;

	return temp;
}

void specify_index(int row, int* n_row_created, int* start_row_index, int div_ratio, int rank, int size, int kernel_len)
{
	if (rank + 1 > size) return;

	//int padded_row_per_side = ((kernel_len - 1) / 2);
	int start = (rank * div_ratio);
	int end = ((rank + 1) * div_ratio) - 1;

	if (rank == size - 1) // last thread takes the rest
	{
		end = row - 1; // index of last row (size of array - size of padding rows)
	}

	int des_row = (end - start + 1) + (kernel_len - 1);


	*n_row_created = des_row; // return the number of created rows
	//n_row_created = &des_row;

	*start_row_index = start; // return the starting index in the padded array
}

int* border_replication(int* img, int img_row, int img_col, int kernel_len)
{
	int* temp = img;
	int row_extended = img_row, col_extended = img_col;
	for (int i = 0; i < (kernel_len - 1)/2; i++)
	{
		temp = padding_2row_2col(temp,row_extended,col_extended);
		row_extended +=2;
		col_extended +=2;
	}
	return temp;
}

int* padding_2row_2col(int* img, int row, int col)
{
	int row_extended = row + 2, col_extended = col + 2;

	int* pading_image = new int[row_extended * col_extended];

	for (int i = 0; i < row_extended; i++)
	{
		for (int j = 0; j < col_extended; j++)
		{
			if (i == 0) // repeate first row
			{
				if (j == 0) // set corner
				{
					pading_image[(i * col_extended) + j] = img[(i * col) + j];
				}
				else if (j == col_extended - 1) // set corner
				{
					pading_image[(i * col_extended) + j] = img[(i * col) + j - 2];
				}
				else // the rest of the cells
				{
					pading_image[(i * col_extended) + j] = img[(i * col) + j - 1];
				}
				continue;
			}
			if (i == row_extended - 1) // repeate last row
			{

				if (j == 0) //set corner
				{
					pading_image[(i * col_extended) + j] = img[((i - 2) * col) + j];

				}
				else if (j == col_extended - 1) // set  corner
				{
					pading_image[(i * col_extended) + j] = img[((i - 2) * col) + j - 2];
				}
				else // the rest of the cells
				{
					pading_image[(i * col_extended) + j] = img[((i - 2) * col) + j - 1];
				}
				continue;
			}
			if ((j == 0 || j == col_extended - 1) && i > 0) // repeate the first and last col
			{
				if (j == 0) // set the first col
				{
					pading_image[(i * col_extended) + j] = img[((i - 1) * col) + j];
				}
				else // set last col
				{
					pading_image[(i * col_extended) + j] = img[((i - 1) * col) + j - 2];
				}
				continue;
			}
			pading_image[(i * col_extended) + j] = img[((i - 1) * col) + j - 1];  // the middle cells
			//pading_image[(i * col_extended) + j] = 0;
		}
	}
	return pading_image;
}

int* portion_image(int* padded_img, int row, int col_extended, int* n_row_created, int div_ratio, int rank, int size, int kernel_len)
{
	int start_index = 0, nu_row_created = 0;

	specify_index(row,&nu_row_created,&start_index,div_ratio,rank,size,kernel_len);

	int* sub_image = new int[nu_row_created * col_extended];

	for (int i = 0; i < nu_row_created; i++)
	{
		for (int j = 0; j < col_extended; j++)
		{
			sub_image[(i * col_extended) + j] = padded_img[start_index];

			start_index++;
		}
	}

	*n_row_created = nu_row_created;

	return sub_image;
}

int* Matrix_mul(int* img, int img_row, int img_col, float* kernel, int kernel_len)  // kernel is float
{
	int* _img = new int[(img_row - (kernel_len - 1)) * (img_col - (kernel_len - 1))];
	int ptr_prev_r, ptr_prev_c;
	float sum = 0;
	int	sum_median = 0;
	int padded_row_per_side = ((kernel_len - 1) / 2);

	float kernel_sum = 0;
	for (int i = 0; i < kernel_len * kernel_len; i++)
	{
		kernel_sum += kernel[i];
	}

	// loop on the main image start from the actual data discarding the padding rows & cols
	for (int i = padded_row_per_side; i < img_row - padded_row_per_side; i++)
	{
		for (int j = padded_row_per_side; j < img_col - padded_row_per_side; j++)
		{
			// get the offset to calculate
			ptr_prev_r = i - (padded_row_per_side); //start from (center index - kernel row) of the img padding 
			ptr_prev_c = j - (padded_row_per_side); //start from (center index - kernel col) of the img padding

			// loop to size of the kernal (apply the kernal from the center)
			for (int i_k = 0; i_k < kernel_len; i_k++)
			{
				for (int j_k = 0; j_k < kernel_len; j_k++)
				{
					//cout << "  img :" << img[((ptr_prev_r + i_k) * img_col) + ptr_prev_c + j_k];
					sum += (float)(img[((ptr_prev_r + i_k) * img_col) + ptr_prev_c + j_k] )* (kernel[(i_k * kernel_len) + j_k]);
				}
				//cout << endl;
			}

			float t = sum / kernel_sum;
			sum_median = ((fmod(t, 1) < 0.5) ? (int)t : (int)t + 1);  // get the median of the multiplication result (result / kernel sum);
			_img[((i - (padded_row_per_side)) * (img_col - (kernel_len - 1))) + (j - (padded_row_per_side))] = sum_median;  // fill the result in the temp result array


			sum = 0; // clear the prev sum & median
			sum_median = 0;
		}
	}
	return _img;
}