#include<mpi.h>
#include <iostream>
//#include<math.h>
using namespace std;

#define mcw MPI_COMM_WORLD

void print(int* img, int start_i, int start_j, int end_row, int end_col, int size_col);
void print(float* img, int start_i, int start_j, int end_row, int end_col, int size_col);
int* padding_2row_2col(int* img, int row, int col);
int* border_replication(int* img, int row, int col, int kernel_len);
int* portion_image(int* img, int row, int col, int* n_row_created, int div_ratio, int rank, int size, int kernel_len);
void specify_index(int row, int* n_row_created, int* start_row_index, int div_ratio, int rank, int size, int kernel_len);
int* Matrix_mul(int* img, int img_row, int img_col, int* kernel, int kernal_len, int kernel_sum);  // kernel is integer
//int* Matrix_mul(int* img, int img_row, int img_col, float* kernel, int kernal_len, float kernel_sum);  // kernel is float
int portion_ratio(int num_row, int size);

int main(int args,char** argv)
{
	 // constant
	//const int row = 7, col = 7;
	//int image[row * col] = {  // row col
	//1,4,2,3,9,5,4, //0
	//2,6,3,2,8,8,3, //1
	//1,4,8,9,3,5,4, //2
	//3,3,5,2,8,1,7, //3
	//5,5,5,7,6,6,5,
	//7,4,9,5,4,5,7,
	//1,7,1,3,1,5,2};

	const int row = 4, col = 5;
	int image[row * col] = {  // row col
	1,4,2,3,9, //0
	2,6,3,2,3, //1
	2,6,3,2,3, //1
	2,6,3,2,3 //1
	};


	//int* rows = new int[7];
	//int* col = new int[7];

	/*const int kernel_len = 3;
	int matrix[kernel_len*kernel_len] = {
		1,1,1,
		1,1,1,
		1,1,1
	};*/
	const int kernel_len = 5;
	int matrix[kernel_len * kernel_len] = {
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1
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

	//int* sub_image ;
	int* sub_image = new int{};
	int sub_img_r = 0;

	int* matrix_mul;

	//  Master Variable
	int Master = 0;
	//int* temp_sub_filtered_img;
	int* filtered_img;


	MPI_Init(NULL, NULL);
	
	int rank=0;
	MPI_Comm_rank(mcw, &rank);

	int size = 0;
	MPI_Comm_size(mcw,&size);

	if (rank == Master)
	{
		

		print(image,0,0, row, col,col);

		cout << "sum" << sum_kernel << endl;
		row_extend = row + (kernel_len - 1);
		col_extend = col + (kernel_len - 1);
		int* pading_image = border_replication(image,row,col,kernel_len);


		print(pading_image,0,0, row_extend, col_extend,col_extend);


		cout << "size:   " << size << endl;
		// divition ratio calcution
		float row_calc_div_f  = (float)row / (float)size;

		if (fmod(row_calc_div_f, 1) >= 0.5)  // fmod return (num of row that make fraction with the divition) 
		{
			row_calc_div_f += 1;
		}
		div_ratio = (int)row_calc_div_f;
		cout << "div_ratio    " << div_ratio << "    mod:    " << fmod(row_calc_div_f, 1) << endl;

		/*cout<<"row_calc_div    "<< row_calc_div_f << "    mod:    " << (row_calc_div_f % 1) << endl;
		if ((row_calc_div_f % 1) >= 0.5)
		{
			row_calc_div_f += 1;
		}
		div_ratio = (int)row_calc_div_f;
		*/
		
		//cout << "rows    " << row_created << endl;
		//row_created = sizeof(splited_image) / (sizeof(int)* col);

		
		
		//sub_image = slice_image_test(pading_image, row_extend, col_extend, &sub_img_r, col_extend, div_ratio,Master, size, kernel_len);

		////print(sub_image,sub_img_r,col_extend);

		MPI_Request request;
		for (int _rank = 0; _rank < size; _rank++)
		{

			int start_row_index = 0;
			int row_created =0;
			specify_index(row, &row_created, &start_row_index, div_ratio, _rank, size, kernel_len);
			
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

				MPI_Send(&row_created,1,MPI_INT,_rank,0,mcw);
				MPI_Send(&pading_image[(start_row_index*col_extend)],row_created*col_extend,MPI_INT,_rank,1,mcw);



				//cout << "rank_send:   " << _rank << "  data:   " << pading_image[start_row_index * col_extend] << endl;
		}
		


		//for (int _rank = 1; _rank < size; _rank++)
		//{
		//	int row_created;
		//	int*_image = slice_image_test(pading_image, row_extend, col_extend, &row_created, col_extend, div_ratio, _rank, size, kernel_len);
		//	
		//	//print(_image, row_created, col_extend);


		//	MPI_Send(&row_created,1,MPI_INT,_rank,0,mcw);
		//	MPI_Send(&_image[0],row_created*col_extend,MPI_INT,_rank,1,mcw);

		//	free(_image);
		//}



		free(pading_image);

		cout << "sub_image_r   " << sub_img_r << endl;

	}
	// global variable
	// broad cast row & col
	MPI_Bcast(&col_extend, 1, MPI_INT, Master, mcw);

	
	// parallel code start

	if (rank != Master)
	{
		MPI_Status stats;
		MPI_Recv(&sub_img_r, 1, MPI_INT, Master, 0, mcw, &stats);
		sub_image = new int[ sub_img_r * col_extend ];
		MPI_Recv(&sub_image[0],sub_img_r*col_extend,MPI_INT,Master,1,mcw,&stats);

		
	}
	
	

	matrix_mul = Matrix_mul(sub_image, sub_img_r, col_extend, matrix, kernel_len, sum_kernel);

	//print(matrix_mul, 0, 0, (sub_img_r - (kernel_len - 1)), (col_extend - (kernel_len - 1)), (col_extend - (kernel_len - 1)));

	free(sub_image);


	if (rank != Master) {
		int row_rank = (sub_img_r - (kernel_len - 1));

		MPI_Send(&row_rank, 1, MPI_INT, Master, Master, mcw);
		MPI_Send(&matrix_mul[0], row_rank * (col_extend - (kernel_len - 1)), MPI_INT, Master, Master + 1, mcw);

		free(matrix_mul);
	}

	// parallel code End
	if (rank == Master)
	{
		filtered_img = new int[row * col];
		int row_rank = 0;
		MPI_Status status;

		//int* result ;
		
		int start_row_index = 0;
		
		for (int _rank = 0; _rank < size; _rank++)
		{
			if (_rank == Master)
			{
				row_rank = (sub_img_r - (kernel_len - 1));  // nu row of master
				//result = matrix_mul;
				
				//cout << "recive:  " << _rank << endl;
				//print(result, row_rank, col_extend - (kernel_len - 1));

				for (int i = start_row_index; i < row_rank; i++)
				{
					for (int j = 0; j < col; j++)
					{
						filtered_img[(i * col) + j] = matrix_mul[(i * col) + j];
					}
				}
				start_row_index += row_rank;
			

				//free(matrix_mul);
				//delete[]result;
				//free(result);

				continue;
			}

			MPI_Recv(&row_rank, 1, MPI_INT, _rank, Master, mcw, &status);

			//result = new int[row_rank * (col_extend - (kernel_len - 1))];
			//MPI_Recv(&result[0], row_rank* (col_extend - (kernel_len - 1)), MPI_INT, _rank, Master + 1, mcw, &status);

			MPI_Recv(&filtered_img[(start_row_index * col)], row_rank* col, MPI_INT, _rank, Master + 1, mcw, &status);
			
			start_row_index += row_rank;

			//cout << "recive:  "<<_rank << endl;
			//print(result, row_rank, col_extend - (kernel_len - 1));

			//delete []result;
			//free(result);
			//MPI_Free_mem(result);
		}
		print(filtered_img ,0,0, row, col,col);

		delete []filtered_img;
	}
	
	MPI_Finalize();
}




// Function Declaretion
void print(int* img, int start_i, int start_j, int end_row, int end_col, int size_col)
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
	cout << "division_ratio for each node  = " << temp << "  row" << endl;

	return temp;
}

int* border_replication(int* img, int row, int col, int kernel_len)
{
	int* temp = img;
	int row_extended = row, col_extended = col;
	for (int i = 0; i < (kernel_len - 1) / 2; i++)
	{
		temp = padding_2row_2col(temp, row_extended, col_extended);
		row_extended += 2;
		col_extended += 2;
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

void specify_index(int row, int* n_row_created, int* start_row_index, int div_ratio, int rank, int size, int kernel_len)
{
	if (rank + 1 > size) return;

	int padded_row_per_side = ((kernel_len - 1) / 2);
	int start = (rank * div_ratio) ;
	int end = ((rank + 1) * div_ratio) - 1 ;

	if (rank == size - 1) // last thread takes the rest
	{
		end = row - 1 ; // index of last row (size of array - size of padding rows)
	}

	int des_row = (end - start + 1) + (kernel_len - 1);


	*n_row_created = des_row; // return the number of created rows
	//n_row_created = &des_row;

	//*start_row_index = start - padded_row_per_side; // return the starting index in the padded array
	*start_row_index = start ; // return the starting index in the padded array
}

int* portion_image(int* img, int row, int col, int* n_row_created, int div_ratio, int rank, int size, int kernel_len)
{
	int start_index = 0, nu_row_created = 0;

	specify_index(row, &nu_row_created, &start_index, div_ratio, rank, size, kernel_len);

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

//int* Matrix_mul(int* img, int img_row, int img_col, int* kernel, int kernel_len,int kernel_sum)
//{
//	int* _img = new int[(img_row - (kernel_len - 1)) * (img_col - (kernel_len - 1))];
//	int ptr_prev_r , ptr_prev_c;
//	int sum = 0,sum_median=0;
//
//	// loop on the main image start from the actual data discarding the padding rows & cols
//	for (int i = kernel_len -2; i < img_row - 1; i++)
//	{
//		for (int j = kernel_len - 2; j < img_col - 1; j++)
//		{
//			// get the offset to calculate
//			ptr_prev_r = i - (kernel_len - 2); //start from (center index - kernel row) of the img padding 
//			ptr_prev_c = j - (kernel_len - 2); //start from (center index - kernel col) of the img padding
//
//			// loop to size of the kernal (apply the kernal from the center)
//			for (int i_k = 0; i_k < kernel_len; i_k++)
//			{
//				for (int j_k = 0; j_k < kernel_len; j_k++)
//				{
//					//cout << "  img :" << img[((ptr_prev_r + i_k) * img_col) + ptr_prev_c + j_k];
//					sum += img[((ptr_prev_r + i_k)* img_col) + ptr_prev_c + j_k] * kernel[(i_k * kernel_len) + j_k];
//				}
//				//cout << endl;
//			}
//
//			float t = (float)sum / (float)kernel_sum;
//			sum_median = ((fmod(t,1) <= 0.5) ? (int)t : (int)t + 1);  // get the median of the multiplication result (result / kernel sum);
//			_img[((i - (kernel_len - 2)) * (img_col - (kernel_len - 1))) + (j - (kernel_len - 2))] = sum_median;  // fill the result in the temp array
//
//
//			sum = 0; // clear the prev sum & median
//			sum_median = 0;
//		}
//	}
//	return _img;
//}

int* Matrix_mul(int* img, int img_row, int img_col, int* kernel, int kernel_len,int kernel_sum) // kernel is integer
{
	int* _img = new int[(img_row - (kernel_len - 1)) * (img_col - (kernel_len - 1))];
	int ptr_prev_r, ptr_prev_c;
	int sum = 0, sum_median = 0;
	int padded_row_per_side = ((kernel_len - 1) / 2);

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

