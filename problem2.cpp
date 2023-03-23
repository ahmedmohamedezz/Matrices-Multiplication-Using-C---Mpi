#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

const int MAX_SIZE = 100;
int main(int argc, char* argv[])
{
	// start mpi
	MPI_Init(&argc, &argv);
	int rank, siz, tag = 0;
	char msg[MAX_SIZE];
	MPI_Status status;
	int r1, c1, r2, c2;
	int eachProcessRows, processes;
	
	// get the process rank && communicator size
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &siz);

	if (rank != 0) {
		// recv rows of first matrix
		// how many elements in the row ?
		MPI_Recv(&r1, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&c1, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
		processes = siz - 1;
		eachProcessRows = r1 / processes;     
		// if the no. of rows is not divisible by the no. of processes : the last process will take the remainig rows ( rank = siz-1 )
		if (rank == siz - 1)
			eachProcessRows += (r1 - eachProcessRows * processes);

		// build matrix
		int** get = (int**)malloc(eachProcessRows * sizeof(int*));
		for (int i = 0; i < eachProcessRows; i++) {
			get[i] = (int*)malloc(c1 * sizeof(int));
		}
		for (int i = 0; i < eachProcessRows; i++)
			MPI_Recv(get[i], c1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

		// recv second matrix
		MPI_Recv(&r2, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&c2, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
		// build the second matrix
		int** mat2 = (int**)malloc(r2 * sizeof(int*));
		for (int i = 0; i < r2; i++)
			mat2[i] = (int*)malloc(c2 * sizeof(int));

		// receive elements
		for (int i = 0; i < r2; i++)
			MPI_Recv(mat2[i], c2, MPI_INT, 0, tag, MPI_COMM_WORLD, &status); // recv row at a time

		// Perform Calculations & send the results back
		for (int i = 0; i < eachProcessRows; i++) {
			// the row to be sent to master process
			int* row = (int*)malloc(c2 * sizeof(int));
			for (int j = 0; j < c2; j++) {
				int res = 0;
				for (int k = 0; k < r2; k++) {
					res += get[i][k] * mat2[k][j];
				}
				row[j] = res;
			}
			MPI_Send(row, c2, MPI_INT, 0, tag, MPI_COMM_WORLD);
			// free
			free(row);
			free(get[i]);
		}
		free(get);

		for (int i = 0; i < r2; i++)
			free(mat2[i]);
		free(mat2);
	}
	else {                       //   Master Process
		printf("Welcome to vector Matrix multiplication program!\n");
		printf("To read dimensions and values from file press 1\n");
		printf("To read dimensions and values from console press 2\n");
		fflush(stdout); // clean the buffer and send the contents to console
		int choise;
		scanf_s("%d", &choise);
		int** mat1, ** mat2;

		if (choise == 1) {
			FILE* inputPtr;
			// read from file the size and the element of each matrix
			// read and initiallize variables ( will fail if the file don't exist )
			fopen_s(&inputPtr,"input.txt", "r");
			// first matrix : read dimension , create matrix , read values
			fscanf_s(inputPtr,"%d %d",&r1,&c1);
			mat1 = (int**)malloc(r1 * sizeof(int*));
			for (int i = 0; i < r1; i++)
				mat1[i] = (int*)malloc(c1 * sizeof(int));
			for (int i = 0; i < r1; i++) {
				for (int j = 0; j < c1; j++) {
					fscanf_s(inputPtr, "%d", &mat1[i][j]);
				}
			}
			// second matrix : read dimension , create matrix , read values
			fscanf_s(inputPtr, "%d %d", &r2, &c2);
			mat2 = (int**)malloc(r2 * sizeof(int*));
			for (int i = 0; i < r2; i++)
				mat2[i] = (int*)malloc(c2 * sizeof(int));
			for (int i = 0; i < r2; i++) {
				for (int j = 0; j < c2; j++) {
					fscanf_s(inputPtr, "%d", &mat2[i][j]);
				}
			}
			// close file
			fclose(inputPtr);
		}
		else {
			printf("Please enter dimensions of the first matrix: \n");
			fflush(stdout); // clean the buffer and send the contents to console
			scanf_s("%d %d", &r1, &c1);
			// allocate memory for the the matrix
			mat1 = (int**)malloc(r1 * sizeof(int*));
			for (int i = 0; i < r1; i++)
				mat1[i] = (int*)malloc(c1 * sizeof(int));

			printf("Please enter its elements: \n");
			fflush(stdout); // clean the buffer and send the contents to console
			for (int i = 0; i < r1; i++) {
				for (int j = 0; j < c1; j++)
					scanf_s("%d", &mat1[i][j]);
			}
			printf("Please enter dimensions of the second matrix: \n");
			fflush(stdout); // clean the buffer and send the contents to console
			scanf_s("%d %d", &r2, &c2);
			// allocate memory for the the matrix
			mat2 = (int**)malloc(r2 * sizeof(int*));
			for (int i = 0; i < r2; i++)
				mat2[i] = (int*)malloc(c2 * sizeof(int));

			printf("Please enter its elements: \n");
			fflush(stdout); // clean the buffer and send the contents to console
			for (int i = 0; i < r2; i++) {
				for (int j = 0; j < c2; j++)
					scanf_s("%d", &mat2[i][j]);
			}
		}

		// check for write dimensions
		if (c1 != r2) {
			printf("The cols of the first matrix must be equal to the rows of the second matrix");
			return 0;
		}
		// in both cases ( file / console ) , send the data
		// send to each process the second matrix & it's row in the first matrix
		// SEND ROW
		// suppose the master process won't participate in calulations ( only handle results )

		for (int i = 1; i < siz; i++) {
			MPI_Send(&r1, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
			MPI_Send(&c1, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
		}

		processes = siz - 1;
		eachProcessRows = r1 / processes; 
		int more = 0;
		int sentRows = eachProcessRows * processes;
		for (int i = 1; i < siz; i++) {
			for (int j = 0; j < (i == siz - 1 ? eachProcessRows + (r1 - sentRows) : eachProcessRows); j++) {
				MPI_Send(mat1[j + more], c1, MPI_INT, i, tag, MPI_COMM_WORLD);     // send c1 values ( complete row )
			}
			more += eachProcessRows;
		}

		// SEND SECOND MATRIX
		for (int i = 1; i < siz; ++i) {
			MPI_Send(&r2, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
			MPI_Send(&c2, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
		}
		// send the second matrix
		for (int i = 1; i < siz; i++) {
			for (int j = 0; j < r2; j++) {
				// send row at a time
				MPI_Send(mat2[j], c2, MPI_INT, i, tag, MPI_COMM_WORLD);
			}
		}

		int** finalMat = (int**)malloc(r1 * sizeof(int*));
		for (int i = 0; i < r1; i++)
			finalMat[i] = (int*)malloc(c2 * sizeof(int));
		// recv results
		more = 0;
		for (int i = 1; i < siz; i++) {
			for (int row = 0; row < (i == siz - 1 ? eachProcessRows + (r1 - sentRows) : eachProcessRows); row++) {
				MPI_Recv(finalMat[row + more], c2, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
			}
			more += eachProcessRows;    // to adjust the index 
		}
		// output
		if (choise != 1) {
			printf("Result Matrix is (%d*%d):\n", r1, c2);
			for (int i = 0; i < r1; i++) {
				for (int j = 0; j < c2; j++)
					printf("%d ", finalMat[i][j]);
				printf("\n");
			}
		}
		else {
			FILE* outputPtr;
			fopen_s(&outputPtr, "output.txt", "w");
			fprintf_s(outputPtr, "Result Matrix is (%d*%d):\n", r1, c2);
			for (int i = 0; i < r1; i++) {
				for (int j = 0; j < c2; j++)
					fprintf_s(outputPtr,"%d ", finalMat[i][j]);
				fprintf_s(outputPtr,"\n");
			}
			// close the file 
			fclose(outputPtr);
		}

		// free matrices
		for (int i = 0; i < r1; i++)
			free(mat1[i]);
		free(mat1);

		for (int i = 0; i < r2; i++)
			free(mat2[i]);
		free(mat2);

		for (int i = 0; i < r1; i++)
			free(finalMat[i]);
		free(finalMat);
	}
	// end mpi
	MPI_Finalize();
}
