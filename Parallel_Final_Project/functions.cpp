#include "main_header.h"

//	Frees allocated memory.
void freeMemory(double** clustersSums, cluster_t* clustersArr, point_t* pointsArr, int totalNumOfClusters) {	
		for(int i = 0; i < omp_get_num_threads() * totalNumOfClusters; i++) 
			free(clustersSums[i]);
		free(clustersSums);
		free(clustersArr);
		free(pointsArr);
}

//	Sends initial settings and the array of points to all slave nodes.
void sendSettings(MPI_Comm Comm, point_t* pointsArr, double settingsArr[], MPI_Datatype pointDatatype, MPI_Datatype clusterDatatype) {
	MPI_Bcast(settingsArr, NUM_OF_INITIAL_SETTINGS, MPI_DOUBLE, MASTER_PROCESS, Comm);
	MPI_Bcast(pointsArr, (int)settingsArr[SET_NUM_OF_POINTS], pointDatatype, MASTER_PROCESS, Comm);
}

//	Receives initial settings and the array of points from the master.
int recvSettings(MPI_Comm Comm, point_t** pointsArr, double settingsArr[], MPI_Datatype pointDatatype, MPI_Datatype clusterDatatype) {		
	int allocationSuccess = TRUE;
	MPI_Bcast(settingsArr, NUM_OF_INITIAL_SETTINGS, MPI_DOUBLE, MASTER_PROCESS, Comm);
	*pointsArr = allocateMainPointsArray((int)settingsArr[SET_NUM_OF_POINTS]);
	if (*pointsArr == NULL)
		return FALSE;
	MPI_Bcast(*pointsArr, (int)settingsArr[SET_NUM_OF_POINTS], pointDatatype, MASTER_PROCESS, Comm);
	return TRUE;
}

//	Sends final results to the master.
void sendFinalResults(MPI_Comm comm, cluster_t* slaveClustersArr, double slaveDistance, int numOfClusters, double slaveT, MPI_Datatype clusterDataType) {	
	MPI_Send(slaveClustersArr, numOfClusters, clusterDataType, MASTER_PROCESS, 0, comm);
	MPI_Send(&slaveDistance, 1, MPI_DOUBLE, MASTER_PROCESS, 0, comm);
	MPI_Send(&slaveT, 1, MPI_DOUBLE, MASTER_PROCESS, 0, comm);
}

//	Receives final results from all slave nodes.
cluster_t* recvFinalResults(MPI_Comm comm, cluster_t* masterClustersArr, double masterDistance, double masterT,		
	double* resultDistance, double* resultT, int numprocs, int numOfClusters, MPI_Status status, MPI_Datatype clusterDatatype) {
	double* tempDistanceArr = (double*)malloc(sizeof(double) * numprocs);
	double* tempTimeArr = (double*)malloc(sizeof(double) * numprocs);
	int chosenNode;
	double tempDistance;
	cluster_t** savedClustersArrs = (cluster_t**)malloc(sizeof(cluster_t*) * numprocs);
	cluster_t* resultClusterArr = (cluster_t*)malloc(sizeof(cluster_t) * numOfClusters);
	for(int i = 0; i < numprocs; i++) {		//	Receives final results from slave nodes.
		savedClustersArrs[i] = (cluster_t*)malloc(sizeof(cluster_t) * numOfClusters);
		if (i == 0) {
			savedClustersArrs[i] = masterClustersArr;
			tempDistanceArr[i] = masterDistance;
			tempTimeArr[i] = masterT;
		}
		else {
			MPI_Recv(savedClustersArrs[i], numOfClusters, clusterDatatype, i, 0, comm, &status);
			MPI_Recv(&(tempDistanceArr[i]), 1, MPI_DOUBLE, i, 0, comm, &status);
			MPI_Recv(&(tempTimeArr[i]), 1, MPI_DOUBLE, i, 0, comm, &status);
		}
	}
	for(int i = 0; i < numprocs; i++) {		//	Compares all results to find the "best" result.
		if (i == 0) {
			chosenNode = 0;
			tempDistance = tempDistanceArr[chosenNode];
		}
		if (tempDistanceArr[i] < tempDistance) {
			chosenNode = i;
			tempDistance = tempDistanceArr[i];
		}
	}
	*resultDistance = tempDistance;
	*resultT = tempTimeArr[chosenNode];
	memcpy(resultClusterArr, savedClustersArrs[chosenNode], sizeof(cluster_t) * numOfClusters);	//	Copies the chosen cluster array to the result array memory space.
	free(tempDistanceArr);
	free(tempTimeArr);
	for(int i = 0; i < numprocs; i++)
		free(savedClustersArrs[i]);
	free(savedClustersArrs);
	return resultClusterArr;
}

//	Creates MPI datatype for the point struct.
MPI_Datatype createPointDatatype() {		
	point_t point;
	MPI_Datatype PointDataType;
	MPI_Datatype pointType[NUM_OF_POINT_FIELDS] = { MPI_INT, MPI_DOUBLE,  MPI_DOUBLE,  MPI_DOUBLE, 
		MPI_DOUBLE,  MPI_DOUBLE, MPI_INT };
	int pointBlockLength[NUM_OF_POINT_FIELDS] = { 1, 1, 1, 1, 1, 1, 1 };
	MPI_Aint pointDisplacement[NUM_OF_POINT_FIELDS];
	pointDisplacement[0] = (char*) &point.id - (char*) &point;
	pointDisplacement[1] = (char*) &point.a - (char*) &point;
	pointDisplacement[2] = (char*) &point.b - (char*) &point;
	pointDisplacement[3] = (char*) &point.x - (char*) &point;
	pointDisplacement[4] = (char*) &point.y - (char*) &point;
	pointDisplacement[5] = (char*) &point.radius - (char*) &point;
	pointDisplacement[6] = (char*) &point.clusterId - (char*) &point;
	MPI_Type_create_struct(NUM_OF_POINT_FIELDS, pointBlockLength, pointDisplacement, pointType, &PointDataType);
	MPI_Type_commit(&PointDataType);
	return PointDataType;
}

//	Creates MPI datatype for the cluster struct.
MPI_Datatype createClusterDatatype() {		
	cluster_t cluster;
	MPI_Datatype ClusterDataType;
	MPI_Datatype clusterType[NUM_OF_CLUSTER_FIELDS] = { MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
	int clusterBlockLength[NUM_OF_CLUSTER_FIELDS] = { 1, 1, 1, 1, 1, 1, 1 };
	MPI_Aint clusterDisplacement[NUM_OF_CLUSTER_FIELDS];
	clusterDisplacement[0] = (char*) &cluster.clusterId - (char*) &cluster;
	clusterDisplacement[1] = (char*) &cluster.x - (char*) &cluster;
	clusterDisplacement[2] = (char*) &cluster.y - (char*) &cluster;
	clusterDisplacement[3] = (char*) &cluster.numOfPoints - (char*) &cluster;
	clusterDisplacement[4] = (char*) &cluster.oldX - (char*) &cluster;
	clusterDisplacement[5] = (char*) &cluster.oldY - (char*) &cluster;
	clusterDisplacement[6] = (char*) &cluster.hasntChanged - (char*) &cluster;
	MPI_Type_create_struct(NUM_OF_CLUSTER_FIELDS, clusterBlockLength, clusterDisplacement, clusterType, &ClusterDataType);
	MPI_Type_commit(&ClusterDataType);
	return ClusterDataType;
}
