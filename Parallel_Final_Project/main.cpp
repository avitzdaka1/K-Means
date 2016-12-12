#include "main_header.h"

int main(int argc, char *argv[])
{
	int namelen, numprocs, myMpiID;	//	MPI properties.
	int successRun = TRUE;		//	"Boolean" variable for a successful run.
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	double t1, t2;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myMpiID);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);	
	MPI_Get_processor_name(processor_name,&namelen);
	MPI_Status status;
	MPI_Datatype pointDataType = createPointDatatype();		//	Create point mpi datatype
	MPI_Datatype clusterDataType = createClusterDatatype();		//	Create cluster mpi datatype
	if (myMpiID == MASTER_PROCESS)
		t1 = MPI_Wtime();
	successRun = calcKMeans(MPI_COMM_WORLD, &status, numprocs, pointDataType, clusterDataType, myMpiID);		//	Enter job function
	if (!successRun)
		printf("An error has occurred, please check previous messages.\n");
	if (myMpiID == MASTER_PROCESS) {
		t2 = MPI_Wtime();
		printf("Time to run = %lf\n", t2 - t1);
	}
	MPI_Finalize();
	return 0;
}