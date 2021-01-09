#include<mpi.h>
#include <iostream>
//#include<math.h>
using namespace std;

#define mcw MPI_COMM_WORLD

void print(int* img, int row, int col);
int* padding(int* img, int row, int col, int row_extended, int col_extended);
int* slice_image(int* img, int row, int col, int* n_row_created, int des_col, int div_ratio,int rank, int size,int kernal_len);

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

	const int kernal_len = 3;
	int matrix[kernal_len][kernal_len] = {
		{1,1,1},
		{1,1,1},
		{1,1,1}
	};


	// global variable
	int div_ratio =0;

	//int* sub_image = new int{};
	int* sub_image;
	int sub_img_r = 0;


	//  Master Variable
	int row_extend=0;
	int col_extend=0;
	int Master = 0;

	MPI_Init(NULL, NULL);
	
	int rank=0;
	MPI_Comm_rank(mcw, &rank);

	int size = 0;
	MPI_Comm_size(mcw,&size);

	if (rank == 0)
	{
		row_extend = row + kernal_len - 1;
		col_extend = col + kernal_len - 1;
		int* pading_image = padding(image,row,col,row_extend,col_extend);


		print(pading_image, row_extend, col_extend);


		float row_calc_div_f = float(row / size);
		if (fmod(row_calc_div_f, size) >= 0.5)
		{
			row_calc_div_f += 1;
		}
		div_ratio = (int)row_calc_div_f;

		

		
		//cout << "rows    " << row_created << endl;
		//row_created = sizeof(splited_image) / (sizeof(int)* col);
		
		int row_created_master;
		int* _image_master = slice_image(pading_image, row_extend, col_extend, &row_created_master, col_extend, div_ratio,0, size, kernal_len);

		sub_img_r = row_created_master;
		sub_image = _image_master;

		free(_image_master);


		for (int _rank = 1; _rank < size; _rank++)
		{
			int row_created;
			int*_image = slice_image(pading_image, row_extend, col_extend, &row_created, col_extend, div_ratio, _rank, size, kernal_len);
			//print(_image, row_created, col_extend);

			cout << "send start    "<<_rank << endl;

			MPI_Send(&row_created,1,MPI_INT,_rank,0,mcw);
			MPI_Send(&_image[0],row_created*col_extend,MPI_INT,_rank,1,mcw);

			cout << "send End     "<<_rank << endl;
			free(_image);
		}

		free(pading_image);
	}
	// global variable
	MPI_Bcast(&div_ratio, 1, MPI_INT, Master, mcw);
	MPI_Bcast(&row_extend, 1, MPI_INT, Master, mcw);
	MPI_Bcast(&col_extend, 1, MPI_INT, Master, mcw);
	
	if (rank != 0)
	{
		cout << "recive start    "<<rank << endl;
		
		MPI_Status stats;
		MPI_Recv(&sub_img_r, 1, MPI_INT, Master, 0, mcw, &stats);
		sub_image = new int[ sub_img_r * col_extend ];
		MPI_Recv(&sub_image[0],sub_img_r*col_extend,MPI_INT,Master,1,mcw,&stats);
		
		
		
		cout << "Rank =  " << rank << " ..." << "row created =    " << sub_img_r << endl;
		print(sub_image, sub_img_r, col_extend);
	
		cout << "recive End    " << rank << endl;
	}
	

	
	
	MPI_Finalize();
}

void print(int* img , int row , int col)
{
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
			if (i == row_extended-1) // repeate last low
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
int* slice_image(int* img, int row, int col,int* n_row_created, int des_col,int div_ratio,int rank,int size,int kernal_len)
{
	int start = (rank * div_ratio) + 1;
	int end = ((rank + 1) * div_ratio) - 1 + 1;

	if (rank == size - 1) // last thread takes the rest
	{
		end = row- (kernal_len - 1); // index of last row
	}

	int des_row = (end - start + 1) + (kernal_len - 1);

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