//#include<iostream>
//#include<mpi.h>
//using namespace std;
//
//#define mcw MPI_COMM_WORLD
//int main(int args, char** argv)
//{
//	MPI_Init(NULL,NULL);
//		int rank=0;
//	MPI_Comm_rank(mcw, &rank);
//
//	int size = 0;
//	MPI_Comm_size(mcw,&size);
//
//	if (rank == 0)
//	{
//		int* arr = new int[400 * 800];
//
//		for (int i = 0; i < 400*800; i++)
//		{
//			arr[i] = 255;
//		}
//
//		MPI_Send(&arr[0],400*800,MPI_INT,1,0,mcw);
//
//		cout << "Data :   " << arr[(400 * 800) - 1] << endl;
//		cout << "   rank:  " << 0 << "  finish" << endl;
//	}
//	if (rank == 1)
//	{
//		int* arr1 = new int[400*800];
//		MPI_Status sta;
//
//		MPI_Recv(&arr1[0],400*800,MPI_INT,0,0,mcw,&sta);
//
//	/*	for (int i = 0; i < 400*800; i++)
//		{
//			cout << arr1[i] << " ";
//		}*/
//
//		cout << endl;
//		cout << "Data :   " << arr1[(400 * 800) - 1] << endl;
//		cout << "finish" << endl;
//	}
//
//	MPI_Finalize();
//}