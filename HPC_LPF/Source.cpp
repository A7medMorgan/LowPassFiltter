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

#define mcw MPI_COMM_WORLD

using namespace std;
using namespace msclr::interop;



void print(int* img, int row, int col);
int* padding(int* img, int row, int col, int row_extended, int col_extended);
int* slice_image(int* img, int row, int col, int* n_row_created, int des_col, int div_ratio, int rank, int size, int kernel_len);
int* Matrix_mul(int* img, int img_row, int img_col, int* kernel, int kernal_len, int kernel_sum);


//int* inputImage(int* w, int* h, System::String^ imagePath) //put the size of image in w & h
//{
//	int* input;
//
//
//	int OriginalImageWidth, OriginalImageHeight;
//
//	//*********************************************************Read Image and save it to local arrayss*************************	
//	//Read Image and save it to local arrayss
//
//	System::Drawing::Bitmap BM(imagePath);
//
//	OriginalImageWidth = BM.Width;
//	OriginalImageHeight = BM.Height;
//	*w = BM.Width;
//	*h = BM.Height;
//	int* Red = new int[BM.Height * BM.Width];
//	int* Green = new int[BM.Height * BM.Width];
//	int* Blue = new int[BM.Height * BM.Width];
//	input = new int[BM.Height * BM.Width];
//	for (int i = 0; i < BM.Height; i++)
//	{
//		for (int j = 0; j < BM.Width; j++)
//		{
//			System::Drawing::Color c = BM.GetPixel(j, i);
//
//			Red[i * BM.Width + j] = c.R;
//			Blue[i * BM.Width + j] = c.B;
//			Green[i * BM.Width + j] = c.G;
//
//			input[i * BM.Width + j] = ((c.R + c.B + c.G) / 3); //gray scale value equals the average of RGB values
//
//		}
//
//	}
//	return input;
//}
//
//
//void createImage(int* image, int width, int height, int index)
//{
//	System::Drawing::Bitmap MyNewImage(width, height);
//
//
//	for (int i = 0; i < MyNewImage.Height; i++)
//	{
//		for (int j = 0; j < MyNewImage.Width; j++)
//		{
//			//i * OriginalImageWidth + j
//			if (image[i * width + j] < 0)
//			{
//				image[i * width + j] = 0;
//			}
//			if (image[i * width + j] > 255)
//			{
//				image[i * width + j] = 255;
//			}
//			System::Drawing::Color c = System::Drawing::Color::FromArgb(image[i * MyNewImage.Width + j], image[i * MyNewImage.Width + j], image[i * MyNewImage.Width + j]);
//			MyNewImage.SetPixel(j, i, c);
//		}
//	}
//	MyNewImage.Save("..//Data//Output//outputRes" + index + ".png");
//	cout << "result Image Saved " << index << endl;
//}


void main(int args, char** argv)
{
	MPI_Init(NULL, NULL);

	int rank;
	MPI_Comm_rank(mcw, &rank);

	int size;
	MPI_Comm_rank(mcw, &size);


	int row = 13, col = 20;
	int* arr = new int[row * col];
	int* rec;

	/*if (rank == 0)
	{*/
	//int ImageWidth = 0, ImageHeight = 0;
	//cout << ImageHeight << "\t" << ImageWidth << endl;
	//int start_s, stop_s, TotalTime = 0;

	//System::String^ imagePath;
	//std::string img;
	//img = "..//Data//Input//test.png";

	////start_s = clock();
	//imagePath = marshal_as<System::String^>(img);
	//int* imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);

	//cout << ImageHeight << "\t" << ImageWidth << endl;

	////print(imageData, ImageWidth, ImageHeight);

	//start_s = clock();
	//createImage(imageData, ImageWidth, ImageHeight, 0);
	//stop_s = clock();
	//TotalTime += (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
	//cout << "time: " << (TotalTime - (TotalTime % 1000)) / 1000<<" Sec" << endl;
	//cout << "actual time: " << TotalTime << endl;
	//free(imageData);

	//cin >> ImageHeight;
	int div_ratio = 0;

	if (rank == 0)
	{
		cout << "row:  " << row << "  size:   " << size << endl;
		//float row_calc_div_f = (float)row / (float)size;
		//cout << "row_calc :   " << row_calc_div_f << endl;
		//if (fmod(row_calc_div_f, 1) >= 0.5)  // fmod return (num of row that make fraction with the divition) 
		//{
		//	row_calc_div_f += 1;
		//}
		//div_ratio = (int)row_calc_div_f;
		//cout << "div_ratio    " << div_ratio << "    mod:    " << fmod(row_calc_div_f, 1) << endl;
	}

	cout << "rank:  " << rank << endl;

	MPI_Bcast(&div_ratio, 1, MPI_INT, 0, mcw);

	/*rec = new int[div_ratio * col];

	MPI_Scatter(&arr,div_ratio,MPI_INT,&rec,div_ratio,MPI_INT,0,mcw);

	print(rec, div_ratio, col);*/
	//}

	MPI_Finalize();

}


void print(int* img, int row, int col)
{
	if (img == NULL) return;

	//MPI_Scatter()
	cout << "row = " << row << "  col = " << col << endl;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			// 2d to 1d array index[(i*col) + j]
			//cout << img[(i * col) + j] << "\t";
			cout << "i " << i << "j:" << j << endl;
		}
		cout << endl;
	}
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
int* slice_image(int* img, int row, int col, int* n_row_created, int des_col, int div_ratio, int rank, int size, int kernel_len)
{
	if (rank + 1 > size) return NULL;

	int start = (rank * div_ratio) + (kernel_len - 2);
	int end = ((rank + 1) * div_ratio) - 1 + (kernel_len - 2);

	if (rank == size - 1) // last thread takes the rest
	{
		end = row - (kernel_len - 1); // index of last row
	}

	int des_row = (end - start + 1) + (kernel_len - 1);

	int* sub_image = new int[(des_row) * (des_col)]; // create the sub image matrix

	*n_row_created = des_row; // return the number of created rows
	//n_row_created = &des_row;


	for (int i = start - 1; i <= start - 1; i++) // take the previouse row
	{
		for (int j = 0; j < des_col; j++)
		{
			// sub_image[0] = img
			sub_image[((i - (start - 1)) * des_col) + j] = img[(i * col) + j];
		}
	}
	for (int i = start; i <= end; i++) // the actule cells and the extended col
	{
		for (int j = 0; j < des_col; j++)
		{
			// index of sub_img is 1 ahead
			sub_image[((i - start + 1) * des_col) + j] = img[(i * col) + j];
		}
	}
	for (int i = end + 1; i <= end + 1; i++) // take the future row
	{
		for (int j = 0; j < des_col; j++)
		{
			// sub_image[des_row + 1] = img[end +1]
			sub_image[((i - (end + 1) + (des_row - 1)) * des_col) + j] = img[(i * col) + j];
		}
	}
	return sub_image;
}
int* Matrix_mul(int* img, int img_row, int img_col, int* kernel, int kernel_len, int kernel_sum)
{
	int* _img = new int[(img_row - (kernel_len - 1)) * (img_col - (kernel_len - 1))];
	int ptr_prev_r, ptr_prev_c;
	int sum = 0, sum_median = 0;

	// loop on the main image start from the actual data discarding the padding rows & cols
	for (int i = kernel_len - 2; i < img_row - 1; i++)
	{
		for (int j = kernel_len - 2; j < img_col - 1; j++)
		{
			// get the offset to calculate
			ptr_prev_r = i - (kernel_len - 2); //start from (center index - kernel row) of the img padding 
			ptr_prev_c = j - (kernel_len - 2); //start from (center index - kernel col) of the img padding

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


			sum_median = ((sum / kernel_sum) % 2 < 0.5) ? (sum / kernel_sum) : ((sum / kernel_sum) + 1);  // get the median of the multiplication result (result / kernel sum);
			_img[((i - (kernel_len - 2)) * (img_col - (kernel_len - 1))) + (j - (kernel_len - 2))] = sum_median;  // fill the result in the temp array


			sum = 0; // clear the prev sum & median
			sum_median = 0;
		}
	}
	return _img;
}


