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
int* padding(int* img, int row, int col, int row_extended, int col_extended);
int* portion_image(int*img, int row, int col, int* n_row_created, int div_ratio, int rank, int size, int kernel_len);
void specify_index(int row, int* n_row_created, int* start_row_index, int div_ratio, int rank, int size, int kernel_len);
int* Matrix_mul(int* img, int img_row, int img_col, int* kernel, int kernal_len, int kernel_sum);
int portion_ratio(int num_row, int size);


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
	cout << "result Image Saved " << index << endl;
}



int main(int args, char** argv)
{
	#pragma region parameter

	// constant
	const int kernel_len = 3;
	int matrix[kernel_len * kernel_len] = {
		1,2,1,
		2,0,2,
		1,2,1
	};

	// global variable
	int div_ratio = 0;
	int row_extend = 0;
	int col_extend = 0;

	int sum_kernel = 0;

	for (int i = 0; i < kernel_len * kernel_len; i++)
	{
		sum_kernel += matrix[i];
	}


	int* sub_image = new int{};
	int sub_img_r = 0;

	// for every thread has its own result
	int* matrix_mul;

	//  Master Variable
	const int Master = 0;
	int row = 0, col = 0; // row , col of the original image
	int* filtered_img;  // the final img
	int start_s = 0, stop_s = 0, TotalTime = 0;

#pragma endregion


	MPI_Init(NULL, NULL);

	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int size = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//cout << "size:    " << size << "   Rank:   " << rank << endl;
	
	if (rank == Master)
	{
		int ImageWidth = 0, ImageHeight = 0;
		
		// get the image
		System::String^ imagePath;
		std::string img;
		img = "..//Data//Input//test.png";
		imagePath = marshal_as<System::String^>(img);

		int* imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);

		row = ImageWidth;
		col = ImageHeight;

		/*row = ImageHeight;
		col = ImageWidth;*/

		//print(imageData,0,0, 3, 10, col);
		


		row_extend = row + (kernel_len - 1); // add the num of row to add on the original array
		col_extend = col + (kernel_len - 1);

		// padding the image
			start_s = clock();  // start the timer
		int* pading_image = padding(imageData, row, col, row_extend, col_extend);


		// divition ratio calcution
				// calculate the ratio of dividing the matrix on every thread equly
		div_ratio = portion_ratio(row,size);

		
		//// split the matrix across the clients
			MPI_Request request;
		for (int _rank = 0; _rank < size; _rank++)
		{

			int start_row_index = 0;
			int row_created = 0;
			specify_index(row_extend, &row_created, &start_row_index, div_ratio, _rank, size, kernel_len);

			cout << "rank:  " << _rank << "  start:    " << start_row_index << "    row_created:" << row_created << endl;

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

			MPI_Send(&row_created, 1, MPI_INT, _rank, Master, mcw);
			cout << "rank_send:   " << _rank << "  data:   " << pading_image[((start_row_index + row_created)* col_extend)-1] << endl;
			//MPI_Send(&pading_image[(start_row_index * col_extend)], row_created * col_extend, MPI_INT, _rank,(Master + 1), mcw);
			
			
			MPI_Isend(&pading_image[(start_row_index * col_extend)], row_created * col_extend, MPI_INT, _rank, (Master + 1), mcw,&request); // do not wait for send complete how ever use recv to wait for data correction
		
		}


		/*for (int _rank = 0; _rank < size; _rank++)
		{
			int start_index = 0;
			int row_created = 0;

			int* _image = new int[1];
			 _image= portion_image(pading_image,row_extend,col_extend,&row_created,div_ratio,_rank,size,kernel_len);

			 if (_rank == Master)
			 {
				 sub_img_r = row_created;
				 sub_image = _image;

				 continue;
			 }

			 cout << "rank_send:   " << _rank << "  data:   " << _image[0] << endl;
			 	MPI_Send(&row_created, 1, MPI_INT, _rank, Master, mcw);
	 			MPI_Send(&_image[0], row_created * col_extend, MPI_INT, _rank,Master + 1, mcw);

				
		}*/
	}

	// global variable
	MPI_Bcast(&col_extend, 1, MPI_INT, Master, mcw);

	// parallel code start

	if (rank != Master)
	{
		MPI_Status stats;
		MPI_Recv(&sub_img_r, 1, MPI_INT, Master, Master, mcw, &stats);

		sub_image = new int[sub_img_r * col_extend];
		MPI_Recv(&sub_image[0], sub_img_r * col_extend, MPI_INT, 0,(Master + 1), mcw, &stats);

		//cout << "rank_recv:   " << rank << "  data:   " << sub_image[(sub_img_r * col_extend) - 1] << endl;


	}
	

	// start the maltiplication of the matrices
	matrix_mul = Matrix_mul(sub_image, sub_img_r, col_extend, matrix, kernel_len, sum_kernel);

	free(sub_image);

	// send back the results to the master
	if (rank != Master) {
		int row_rank = (sub_img_r - (kernel_len - 1)); // send back only the needed data and discard the padded row&col

		MPI_Send(&row_rank, 1, MPI_INT, Master, Master, mcw);
		MPI_Send(&matrix_mul[0], row_rank * (col_extend - (kernel_len - 1)), MPI_INT, Master, (Master + 1), mcw);

		free(matrix_mul);
	}


	// parallel code End

	// recv back the result 
	if (rank == Master)
	{
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

				//free(matrix_mul);

				continue;
			}

			MPI_Recv(&row_rank, 1, MPI_INT, _rank, Master, mcw, &status);

			//cout << "  Rank:  " << _rank << "  Receve row_rank" << row_rank << endl;

			MPI_Recv(&filtered_img[(start_row_index * col)], row_rank * col, MPI_INT, _rank, (Master + 1), mcw, &status);

			start_row_index += row_rank;

		}
			
			stop_s = clock();
		// create the image

			cout << "the image will be save as ResOutput(n) ::" << "Enter number replace (n)" << "\t";
			int n = 0;
			cin >> n;
		createImage(filtered_img, row, col, n);

			print(filtered_img,0,0,3,10,col);

		TotalTime += (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
		cout << "time: " << TotalTime << endl;


		delete[]filtered_img;
	}

	MPI_Finalize();
}
void print(int* img,int start_i,int start_j, int end_row, int end_col,int size_col)
{
	if (img == NULL) return;

	cout << "row = " << end_row << "  col = " << end_col << endl;
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
	cout << "div_ratio    " << temp << "    mod:    " << fmod(row_calc_div_f, 1) << endl;

	return temp;
}

int* padding(int* img, int row, int col, int row_extended, int col_extended)
{
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

void specify_index(int row, int* n_row_created, int* start_row_index, int div_ratio, int rank, int size, int kernel_len)
{
	if (rank + 1 > size) return;

	int padded_row_per_side = ((kernel_len - 1) / 2);
	int start = (rank * div_ratio) + padded_row_per_side;
	int end = ((rank + 1) * div_ratio) - 1 + padded_row_per_side;

	if (rank == size - 1) // last thread takes the rest
	{
		end = row - (kernel_len - 1); // index of last row (size of array - size of padding rows)
	}

	int des_row = (end - start + 1) + (kernel_len - 1);


	*n_row_created = des_row; // return the number of created rows
	//n_row_created = &des_row;

	*start_row_index = start - padded_row_per_side; // return the starting index in the padded array
}

int* portion_image(int* img, int row, int col, int* n_row_created, int div_ratio, int rank, int size, int kernel_len)
{
	int start_index = 0, nu_row_created = 0;

	specify_index(row,&nu_row_created,&start_index,div_ratio,rank,size,kernel_len);

	int* sub_image = new int[nu_row_created * col];

	for (int i = 0; i < nu_row_created; i++)
	{
		for (int j = 0; j < col; j++)
		{
			sub_image[(i * col) + j] = img[start_index];

			start_index++;
		}
	}

	*n_row_created = nu_row_created;

	return sub_image;
}

int* Matrix_mul(int* img, int img_row, int img_col, int* kernel, int kernel_len, int kernel_sum)
{
	int* _img = new int[(img_row - (kernel_len - 1)) * (img_col - (kernel_len - 1))];
	int ptr_prev_r, ptr_prev_c;
	int sum = 0, sum_median = 0;
	int padded_row_per_side = ((kernel_len - 1) / 2);

	// loop on the main image start from the actual data discarding the padding rows & cols
	for (int i = padded_row_per_side; i < img_row - 1; i++)
	{
		for (int j = padded_row_per_side; j < img_col - 1; j++)
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
					sum += img[((ptr_prev_r + i_k) * img_col) + ptr_prev_c + j_k] * kernel[(i_k * kernel_len) + j_k];
				}
				//cout << endl;
			}

			float t = (float)sum / (float)kernel_sum;
			sum_median = ((fmod(t, 1) < 0.5) ? (int)t : (int)t + 1);  // get the median of the multiplication result (result / kernel sum);
			_img[((i - (padded_row_per_side)) * (img_col - (kernel_len - 1))) + (j - (padded_row_per_side))] = sum_median;  // fill the result in the temp result array


			sum = 0; // clear the prev sum & median
			sum_median = 0;
		}
	}
	return _img;
}
