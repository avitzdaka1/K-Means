#include "main_header.h"

//	Main job code.
int calcKMeans(MPI_Comm Comm, MPI_Status* status, int numprocs,	MPI_Datatype pointDatatype, MPI_Datatype clusterDatatype, int myMpiID) {
	double settingsArr[NUM_OF_INITIAL_SETTINGS];		//	Initial settings array (from first line of input file).
	point_t* pointsArr;			//	The points array.
	cluster_t* clustersArr;		//	The clusters array.
	double** clustersSums;
	double minimalT, minimalDistance;	//	Minimal T (time) and minimal distance for later use.
	if (myMpiID == MASTER_PROCESS) {
		int fileReadSuccess = TRUE;	//	"Boolean" variable for the file reading function.
		fileReadSuccess = readFile(settingsArr, &pointsArr, &clustersArr);
		if (!fileReadSuccess) { 	//	If the input file reading function has encountered a problem, 
			printf("There was an error reading from input file, program will now exit.");		//	return FALSE.
			return FALSE;
		}
		sendSettings(Comm, pointsArr, settingsArr, pointDatatype, clusterDatatype);		//	Sends initial settings to all other nodes.
	}
	else 
		recvSettings(Comm, &pointsArr, settingsArr, pointDatatype, clusterDatatype);	//	Receives settings from master (run by slaves).
	int allocationSuccess = allocateClusters(settingsArr, &clustersArr);	//	Allocates memory for the clusters.
	clustersSums = (double**) malloc(sizeof(double*) * omp_get_num_procs() * (int)settingsArr[SET_NUM_OF_CLUSTERS]);
	for(int i = 0; i < (omp_get_num_procs() * settingsArr[SET_NUM_OF_CLUSTERS]); i++) {
		clustersSums[i] = (double*) calloc(3, sizeof(double));
	}
	if (!allocationSuccess && myMpiID == MASTER_PROCESS) {		//	If allocation did not succeed, return FALSE.
		printf("There was an error allocating memory for the clusters array or the clusters' points array, program will now exit.\n");
		return FALSE;
	}
	cluster_t* savedClustersArr = (cluster_t*)malloc((int)settingsArr[SET_NUM_OF_CLUSTERS]*sizeof(cluster_t));	//	Allocates memory for the clusters that 
																										//    will be found during the algorithm run.
	allocatePointsCuda(pointsArr, (int)settingsArr[SET_NUM_OF_POINTS]);
	TCalc(myMpiID, settingsArr, numprocs, pointsArr, clustersArr,		//	Code sequence of the actual algorithm calculation.
	 &minimalDistance, &minimalT, savedClustersArr, clustersSums);
	if (myMpiID == MASTER_PROCESS)
		masterFinalCode((int)settingsArr[SET_NUM_OF_CLUSTERS], Comm, savedClustersArr, minimalDistance, minimalT, numprocs,
		status, clusterDatatype, clustersSums, clustersArr, pointsArr);	//	Final sequence of code to be run by the master process
	else {																						//	at the end of the calculation (receives results and writes them to file).
		sendFinalResults(Comm, savedClustersArr, minimalDistance, (int)settingsArr[SET_NUM_OF_CLUSTERS], minimalT, clusterDatatype);	// Sends results to the master
		freeMemory(clustersSums, clustersArr, pointsArr, (int)settingsArr[SET_NUM_OF_CLUSTERS]);					//	(used by the slave nodes), and clears the memory that was allocated. 
	}
	return TRUE;
}

//	Code sequence of the actual algorithm calculation.
void TCalc(int myMpiID, double settingsArr[], int numprocs, point_t* pointsArr, cluster_t* clustersArr,		
	 double* minimalDistance, double* minimalT, cluster_t* savedClustersArr, double** clustersSums) {
		double tempDistance, currentT;		//	Temp minimal distance variable.
		int clustersChanged, hasChanged;
	for	(currentT = (myMpiID * settingsArr[SET_DELTA_T]); currentT < settingsArr[SET_TIME_INTERVAL]; currentT += (settingsArr[SET_DELTA_T] * numprocs)) {
		clustersChanged = 0;
		hasChanged = TRUE;	//	"Boolean" variable to check if no cluster centers were changed during the last iteration.
		pointRelocation(pointsArr, settingsArr[SET_TIME_INTERVAL], currentT, (int)settingsArr[SET_NUM_OF_POINTS]);	//	Relocates the points, each one on their own circle.
		createClusters(pointsArr, clustersArr, (int)settingsArr[SET_NUM_OF_CLUSTERS]);	//	Creates clusters from the K first points in the recently changed points array.
		for(int i = 0; i < settingsArr[SET_ITERATIONS_LIMIT]; i++) {		//	Calculates the clusters untill X iterations occured or the calculation has ended.
			assignToCluster(pointsArr, clustersArr, (int)settingsArr[SET_NUM_OF_CLUSTERS], (int)settingsArr[SET_NUM_OF_POINTS], clustersSums);	//	Calculates distances from points to clusters.
			hasChanged = calcNewClustCenter(clustersSums, clustersArr, pointsArr, (int)settingsArr[SET_NUM_OF_CLUSTERS], (int)settingsArr[SET_NUM_OF_POINTS], &clustersChanged);	//	Calculates the average center for each cluster.
			if (!hasChanged) 	//	If all cluster centers remained the same since the last iteration, break.
				break;
		}
		deterMinClusterDistance(clustersArr, &tempDistance, (int)settingsArr[SET_NUM_OF_CLUSTERS]);		//	Determines the minimal distance between two clusters.
		if (currentT == myMpiID * settingsArr[SET_DELTA_T]) {	//	If this is the first T iteration for a given computer node.
			*minimalDistance = tempDistance;
			*minimalT = currentT;
			saveClustersState(clustersArr, savedClustersArr, (int)settingsArr[SET_NUM_OF_CLUSTERS]);	//	Save all cluster centers at this current T time.
		}
		if (tempDistance < *minimalDistance) {	//	If a minimal distance was found, save clusters and T time.
			*minimalDistance = tempDistance;
			*minimalT = currentT;
			saveClustersState(clustersArr, savedClustersArr, (int)settingsArr[SET_NUM_OF_CLUSTERS]);
		}
	}
}

//	Calculates new center for each cluster.
int calcNewClustCenter(double** clustersSums, cluster_t* clusterArr, point_t* pointsArr, int totalNumOfClusters, int totalNumOfPoints, int* clustersUnChanged) {
		int i, j;
		double sumX, sumY;
#pragma omp parallel for private(i, j, sumX, sumY)
		for(i = 0; i < totalNumOfClusters; i++) {
			clusterArr[i].oldX = clusterArr[i].x;
			clusterArr[i].oldY = clusterArr[i].y;
			sumX = 0;
			sumY = 0;
			for(j = 0; j < omp_get_num_threads(); j++) {
				sumX += clustersSums[(j * totalNumOfClusters) + i][CLUSTER_SUM_X];
				sumY += clustersSums[(j * totalNumOfClusters) + i][CLUSTER_SUM_Y];
				clusterArr[i].numOfPoints += (int)clustersSums[(j * totalNumOfClusters) + i][CLUSTER_SUM_NUM_OF_POINTS];
			}
			if (clusterArr[i].numOfPoints > 0) {
				clusterArr[i].x = sumX / clusterArr[i].numOfPoints;
				clusterArr[i].y = sumY / clusterArr[i].numOfPoints;
			}
			clusterArr[i].numOfPoints = 0;
			//	Checks if the cluster's center hasn't changed and if the hasntChanged flag is off.
			if (clusterArr[i].x == clusterArr[i].oldX && clusterArr[i].y == clusterArr[i].oldY && clusterArr[i].hasntChanged == FALSE) {
				clusterArr[i].hasntChanged = TRUE;
				(*clustersUnChanged)++;
			}
			//	Clears the clustesSums data
			for(j = 0; j < omp_get_num_threads(); j++) {
				clustersSums[(j * totalNumOfClusters) + i][CLUSTER_SUM_X] = 0;
				clustersSums[(j * totalNumOfClusters) + i][CLUSTER_SUM_Y] = 0;
				clustersSums[(j * totalNumOfClusters) + i][CLUSTER_SUM_NUM_OF_POINTS] = 0;
			}
		}
	if (*clustersUnChanged == totalNumOfClusters)
		return FALSE;
	return TRUE;
}

//	Relocates all points over their circle.
void pointRelocation(point_t* pointsArr, double timeInterval, double currentT, int totalNumOfPoints) {		
	double cosT = cos((2 * PI * currentT) / timeInterval), sinT = sin((2 * PI * currentT) / timeInterval);
	pointRelocationCuda(pointsArr, timeInterval, currentT, totalNumOfPoints, cosT, sinT);
}

//	Determines what cluster the given point belongs to.
void assignToCluster(point_t* pointArr,	cluster_t* clusterArr, int totalNumOfClusters, int totalNumOfPoints, double** clustersSums) {
		int i, j, clusterChosen;
		double tempDistance, minDistance;
#pragma omp parallel for private (i, j, clusterChosen, tempDistance, minDistance)
		for(j = 0; j < totalNumOfPoints; j++) {
			minDistance = INT_MAX;
			for(i = 0; i < totalNumOfClusters; i++) {		//	Checks all distances from the given point to all clusters.
				tempDistance = calcDistance(pointArr[j].x, pointArr[j].y, clusterArr[i].x, clusterArr[i].y);
				if (tempDistance < minDistance)	{	//	If current distance is smaller than the minimum, change it to be the new minimum distance.
					clusterChosen = clusterArr[i].clusterId;
					minDistance = tempDistance;
				}
			}
			pointArr[j].clusterId = clusterChosen;
			//	"Adds" the point to the cluster, so a new cluster center can be calculated later.
			clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointArr[j].clusterId][CLUSTER_SUM_X] += pointArr[j].x;
			clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointArr[j].clusterId][CLUSTER_SUM_Y] += pointArr[j].y;
			clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointArr[j].clusterId][CLUSTER_SUM_NUM_OF_POINTS] += 1;
		}
}

//	Calculates the distance between two XY points.
double calcDistance(double pointAX, double pointAY, double pointBX, double pointBY) {		
	return sqrt(pow((pointAX - pointBX), 2) + pow((pointAY - pointBY), 2));
}

//	Calculates the minimal distance between two clusters.
void deterMinClusterDistance(cluster_t *clusterArr, double* distance, int totalNumOfClusters) {	
	double tempDistance, currentMinimum = calcDistance(clusterArr[0].x, clusterArr[0].y, clusterArr[1].x, clusterArr[1].y);
	int i, j;
#pragma omp parallel for private (i, j)
	for(i = 0; i < totalNumOfClusters; i++) {
		for(j = i+1; j < totalNumOfClusters; j++) {
			tempDistance = calcDistance(clusterArr[i].x, clusterArr[i].y, clusterArr[j].x, clusterArr[j].y);	//	Calculates distance between two current clusters.
			if (tempDistance < currentMinimum)
				currentMinimum = tempDistance;
		}
	}
	*distance = currentMinimum;
}

//	Creates clusters using the first K points coordinates.
void createClusters(point_t* pointsArr, cluster_t* clusterArr, int totalNumOfClusters) {
	int i;
#pragma omp parallel for private(i)
	for(i = 0; i < totalNumOfClusters; i++) {
		clusterArr[i].clusterId = i;
		clusterArr[i].numOfPoints = 0;
		clusterArr[i].x = pointsArr[i].x;
		clusterArr[i].y = pointsArr[i].y;
		clusterArr[i].hasntChanged = FALSE;
	}
}

//	Saves clusters state.
void saveClustersState(cluster_t* clustersArr, cluster_t* savedClustersArr, int totalNumOfClusters) {		
	int i;
#pragma omp parallel for private (i)
	for(i = 0; i < totalNumOfClusters; i++) {
		savedClustersArr[i].clusterId = clustersArr[i].clusterId;
		savedClustersArr[i].x = clustersArr[i].x;
		savedClustersArr[i].y = clustersArr[i].y;
	}
}

//	Allocates an array of point_t types for the points being read from the file.
point_t* allocateMainPointsArray(int totalNumOfPoints) {		
	point_t* pointsArr = (point_t*)malloc(sizeof(point_t) * totalNumOfPoints);
	if (pointsArr == NULL)
		printf("Error allocating memory for the points array.\n");
	return pointsArr;
}

//	Allocates memory for the clusters array.
int allocateClusters(double settingsArr[],		
	cluster_t** clusterArr) {		//	clusters' points array.
	*clusterArr = (cluster_t*)malloc(sizeof(cluster_t) * (int)settingsArr[SET_NUM_OF_CLUSTERS]);
	if (clusterArr == NULL) {
		printf("Error allocating memory for the clusters array.\n");
		return FALSE;
	}
	return TRUE;
}

//	Master's final sequence of code.
void masterFinalCode(int totalNumOfClusters, MPI_Comm Comm, cluster_t* savedClustersArr, double minimalDistance, double minimalT,		
	int numprocs, MPI_Status *status, MPI_Datatype clusterDatatype, double** clustersSums, cluster_t* clustersArr, point_t* pointsArr) {
	cluster_t* finalClusterArr = (cluster_t*)malloc(sizeof(cluster_t) * totalNumOfClusters);	//	Final cluster array to be saved to the output file.
		double finalDistance, finalT;		//	Final distance and time T to be saved to the output file.
		finalClusterArr = recvFinalResults(Comm, savedClustersArr, minimalDistance, minimalT, &finalDistance, // Receives final results from slave nodes.
			&finalT, numprocs, totalNumOfClusters, *status, clusterDatatype); 
		writeToFile(finalDistance, finalT, finalClusterArr, totalNumOfClusters);	//	Writes results to the output file.
		freeMemory(clustersSums, clustersArr, pointsArr, totalNumOfClusters);	//	Frees the memory allocated by the master.
}