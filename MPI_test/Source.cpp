#include<mpi.h>
#include <iostream>
//#include<math.h>
using namespace std;

#define mcw MPI_COMM_WORLD

void print(int* img, int row, int col);
int* padding(int* img, int row, int col, int row_extended, int col_extended);
int* slice_image(int* img, int row, int col, int* n_row_created, int des_col, int div_ratio,int rank, int size,int kernel_len);
int* Matrix_mul(int* img, int img_row, int img_col, int* kernel, int kernal_len, int kernel_sum);

int main(int args,char** argv)
{
	 // constant
	const int row = 7, col = 7;
	int image[row * col] = {  // row col
	1,4,2,3,2,5,4, //0
	2,4,3,2,2,8,3, //1
	1,4,8,9,2,5,4, //2
	3,4,5,2,4,1,7, //3
	5,4,5,7,2,6,5,
	7,4,9,5,2,5,8,
	1,4,1,3,2,5,2 };
	//int* rows = new int[7];
	//int* col = new int[7];

	const int kernel_len = 3;
	int matrix[kernel_len*kernel_len] = {
		1,1,1,
		1,1,1,
		1,1,1
	};


	// global variable
	int div_ratio =0;
	int row_extend = 0;
	int col_extend = 0;

	int sum_kernel = 0;

	for (int i = 0; i < kernel_len*kernel_len; i++)
	{
		sum_kernel += matrix[i];
	}

	//int* sub_image = new int{};
	int* sub_image = new int{};
	int sub_img_r = 0;

	int* matrix_mul;

	//  Master Variable
	int Master = 0;
	int* temp_sub_filtered_img;
	int* filtered_img;


	MPI_Init(NULL, NULL);
	
	int rank=0;
	MPI_Comm_rank(mcw, &rank);

	int size = 0;
	MPI_Comm_size(mcw,&size);

	if (rank == 0)
	{

		print(image, row, col);

		cout << "size" << size << endl;
		row_extend = row + (kernel_len - 1);
		col_extend = col + (kernel_len - 1);
		int* pading_image = padding(image,row,col,row_extend,col_extend);


		print(pading_image, row_extend, col_extend);


		float row_calc_div_f  = (float)row / (float)size;
		

		/*cout<<"row_calc_div    "<< row_calc_div_f << "    mod:    " << (row_calc_div_f % 1) << endl;
		if ((row_calc_div_f % 1) >= 0.5)
		{
			row_calc_div_f += 1;
		}
		div_ratio = (int)row_calc_div_f;
		*/


		// divition ratio calcution

		//cout << "row_calc_div    " << row_calc_div_f << "    mod:    " << fmod(row_calc_div_f, 1) << endl;
		if (fmod(row_calc_div_f, 1) >= 0.5)  // fmod return (num of row that make fraction with the divition) 
		{
			row_calc_div_f += 1;
		}
		div_ratio = (int)row_calc_div_f;


		
		cout << "ratio :  " << div_ratio << endl;
		
		//cout << "rows    " << row_created << endl;
		//row_created = sizeof(splited_image) / (sizeof(int)* col);
		
		sub_image = slice_image(pading_image, row_extend, col_extend, &sub_img_r, col_extend, div_ratio,Master, size, kernel_len);

		print(sub_image,sub_img_r,col_extend);


		



		for (int _rank = 1; _rank < size; _rank++)
		{
			int row_created;
			int*_image = slice_image(pading_image, row_extend, col_extend, &row_created, col_extend, div_ratio, _rank, size, kernel_len);
			
			print(_image, row_created, col_extend);


			MPI_Send(&row_created,1,MPI_INT,_rank,0,mcw);
			MPI_Send(&_image[0],row_created*col_extend,MPI_INT,_rank,1,mcw);

			free(_image);
		}

		free(pading_image);
	}
	// global variable

	MPI_Bcast(&col_extend, 1, MPI_INT, Master, mcw);
	
	// parallel code start

	if (rank != 0)
	{
		//cout << "recive start    "<<rank << endl;
		
		MPI_Status stats;
		MPI_Recv(&sub_img_r, 1, MPI_INT, Master, 0, mcw, &stats);
		sub_image = new int[ sub_img_r * col_extend ];
		MPI_Recv(&sub_image[0],sub_img_r*col_extend,MPI_INT,Master,1,mcw,&stats);
		
		
		/*
		cout << "Rank =  " << rank << " ..." << "row created =    " << sub_img_r << endl;
		print(sub_image, sub_img_r, col_extend);
	
		cout << "recive End    " << rank << endl;*/
	}
	
	

	matrix_mul = Matrix_mul(sub_image, sub_img_r, col_extend, matrix, kernel_len, sum_kernel);

	//print(matrix_mul, sub_img_r - (kernel_len - 1), col_extend - (kernel_len - 1));

	if (rank != 0) {
		//cout << "send result" << endl;
		int row_rank = (sub_img_r - (kernel_len - 1));

		MPI_Send(&row_rank, 1, MPI_INT, Master, Master, mcw);
		MPI_Send(&matrix_mul[0], row_rank * (col_extend - (kernel_len - 1)), MPI_INT, Master, Master + 1, mcw);
	}

	// parallel code End
	if (rank == 0)
	{
		cout << "size" << size << endl;
		filtered_img = new int[row * col];
		int row_rank = 0;
		MPI_Status status;

		int* result ;
		for (int _rank = 0; _rank < size; _rank++)
		{
			if (_rank == 0)
			{
				row_rank = (sub_img_r - (kernel_len - 1));
				result = matrix_mul;
				cout << "recive:  " << _rank << endl;
				print(result, row_rank, col_extend - (kernel_len - 1));

				continue;
			}

			MPI_Recv(&row_rank, 1, MPI_INT, _rank, Master, mcw, &status);

			result = new int[row_rank * (col_extend - (kernel_len - 1))];
			MPI_Recv(&result[0], row_rank * (col_extend - (kernel_len - 1)), MPI_INT, _rank, Master + 1, mcw, &status);


			cout << "recive:  "<<_rank << endl;
			print(result, row_rank, col_extend - (kernel_len - 1));

			free(result);
			//MPI_Free_mem(result);
		}
		
	}
	
	MPI_Finalize();
}

void print(int* img , int row , int col)
{
	if (img == NULL) return;

	cout << "row = " << row << "  col = " << col << endl;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			// 2d to 1d array index[(i*col) + j]
			cout << img[(i*col )+ j] << "\t";
		}
		cout << endl;
	}
}
int* padding(int* img, int row, int col, int row_extended, int col_extended)
{
	int* pading_image = new int[ row_extended * col_extended ];

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
			if (i == row_extended-1) // repeate last row
			{
				
				if (j == 0) //set corner
				{
					pading_image[(i * col_extended) + j] = img[((i - 2) * col) + j];
					
				}
				else if (j == col_extended - 1) // set  corner
				{
					pading_image[(i * col_extended) + j] = img[((i - 2 )* col) + j - 2];
				}
				else // the rest of the cells
				{
					pading_image[(i * col_extended) + j] = img[((i - 2) * col) + j - 1];
				}
				continue;
			}
			if ((j == 0 || j == col_extended - 1)&& i > 0 ) // repeate the first and last col
			{
				if (j==0) // set the first col
				{
					pading_image[(i * col_extended) + j] = img[((i - 1) * col) + j];
				}
				else // set last col
				{
					pading_image[(i * col_extended) + j] = img[((i - 1) * col) + j - 2];
				}
				continue;
			}
			pading_image[(i * col_extended) + j] = img[((i - 1 )* col) + j - 1];  // the middle cells
			//pading_image[(i * col_extended) + j] = 0;
		}	
	}
	return pading_image;
}
int* slice_image(int* img, int row, int col,int* n_row_created, int des_col,int div_ratio,int rank,int size,int kernel_len)
{
	if (rank + 1 > size) return NULL;

	int start = (rank * div_ratio) + (kernel_len - 2);
	int end = ((rank + 1) * div_ratio) - 1 + (kernel_len - 2);

	if (rank == size - 1) // last thread takes the rest
	{
		end = row- (kernel_len - 1); // index of last row
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
			sub_image[((i - (end + 1)+(des_row - 1)) * des_col) + j] = img[(i * col) + j];
		}
	}
	return sub_image;
}
int* Matrix_mul(int* img, int img_row, int img_col, int* kernel, int kernel_len,int kernel_sum)
{
	int* _img = new int[(img_row - (kernel_len - 1)) * (img_col - (kernel_len - 1))];
	int ptr_prev_r , ptr_prev_c;
	int sum = 0,sum_median=0;

	// loop on the main image start from the actual data discarding the padding rows & cols
	for (int i = kernel_len -2; i < img_row - 1; i++)
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
					sum += img[((ptr_prev_r + i_k)* img_col) + ptr_prev_c + j_k] * kernel[(i_k * kernel_len) + j_k];
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