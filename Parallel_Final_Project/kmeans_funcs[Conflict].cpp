#include "main_header.h"

int calcKMeans(MPI_Comm Comm, MPI_Status* status, int numprocs,		//	Main job code.
	MPI_Datatype pointDatatype, MPI_Datatype clusterDatatype, int myMpiID) {
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

void TCalc(int myMpiID, double settingsArr[], int numprocs, point_t* pointsArr, cluster_t* clustersArr,		//	Code sequence of the actual algorithm calculation.
	 double* minimalDistance, double* minimalT, cluster_t* savedClustersArr, double** clustersSums) {
		double tempDistance;		//	Temp minimal distance variable.
	for	(double currentT = (myMpiID * settingsArr[SET_DELTA_T]); currentT < settingsArr[SET_TIME_INTERVAL]; currentT += (settingsArr[SET_DELTA_T] * numprocs)) {
		int clustersChanged = 0;
		int hasChanged = TRUE;	//	"Boolean" variable to check if no cluster centers were changed during the last iteration.
		pointRelocation(pointsArr, settingsArr[SET_TIME_INTERVAL], currentT, (int)settingsArr[SET_NUM_OF_POINTS]);	//	Relocates the points, each one on their own circle.
		createClusters(pointsArr, clustersArr, (int)settingsArr[SET_NUM_OF_CLUSTERS]);	//	Creates clusters from the K first points in the recently changed points array.
		for(int i = 0; i < settingsArr[SET_ITERATIONS_LIMIT]; i++) {		//	Calculates the clusters untill X iterations occured or the calculation has ended.
			assignToCluster(pointsArr, clustersArr, (int)settingsArr[SET_NUM_OF_CLUSTERS], (int)settingsArr[SET_NUM_OF_POINTS]);	//	Calculates distances from points to clusters.
			hasChanged = calcNewClustCenter(clustersSums, clustersArr, pointsArr, (int)settingsArr[SET_NUM_OF_CLUSTERS], (int)settingsArr[SET_NUM_OF_POINTS], &clustersChanged);	//	Calculates the average center for each cluster.
			if (!hasChanged) {	//	If all cluster centers remained the same since the last iteration, break.
				//printf("iteration number = %d, t = %lf\n", i, currentT);
			//	fflush(stdout);
				break;
			}
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

int calcNewClustCenter(double** clustersSums, cluster_t* clusterArr, point_t* pointsArr,		//	Calculates new center for each cluster.
	int totalNumOfClusters, int totalNumOfPoints, int* clustersUnChanged) {
#pragma omp parallel for
		for(int i = 0; i < totalNumOfPoints; i++) {
				clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointsArr[i].clusterId][0] += pointsArr[i].x;
				clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointsArr[i].clusterId][1] += pointsArr[i].y;
				clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointsArr[i].clusterId][2] += 1;
				//printf("new x = %lf, new y = %lf, num of points = %lf\n", clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointsArr[i].clusterId][0], clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointsArr[i].clusterId][1], clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointsArr[i].clusterId][2]);
		}
#pragma omp parallel for 
		for(int i = 0; i < totalNumOfClusters; i++) {
			clusterArr[i].oldX = clusterArr[i].x;
			clusterArr[i].oldY = clusterArr[i].y;
			double sumX = 0;
			double sumY = 0;
			for(int j = 0; j < omp_get_num_threads(); j++) {
				sumX += clustersSums[(j * totalNumOfClusters) + i][0];
				sumY += clustersSums[(j * totalNumOfClusters) + i][1];
				clusterArr[i].numOfPoints += (int)clustersSums[(j * totalNumOfClusters) + i][2];
				//printf("i = %d, num of points = %d\n", i, clusterArr[i].numOfPoints);
				//printf("i = %d, j = %d, sumx = %lf, sumy = %lf, num of points = %d\n", i, j, sumX, sumY, clusterArr[i].numOfPoints);
			}
			if (clusterArr[i].numOfPoints > 0) {
				clusterArr[i].x = sumX / clusterArr[i].numOfPoints;
				clusterArr[i].y = sumY / clusterArr[i].numOfPoints;
				//printf("i = %d, new x = %lf, new y = %lf\n", i, clusterArr[i].x, clusterArr[i].y);
			}
			clusterArr[i].numOfPoints = 0;
			if (clusterArr[i].x == clusterArr[i].oldX && clusterArr[i].y == clusterArr[i].oldY && clusterArr[i].hasntChanged == FALSE) {
				clusterArr[i].hasntChanged = TRUE;
#pragma omp critical
			{
				(*clustersUnChanged)++;
			}
			}
		}
		//printf("unchanged clusters = %d\n", *clustersUnChanged);
#pragma omp parallel for
		for(int i = 0; i < totalNumOfPoints; i++) {
			for(int j = 0; j < totalNumOfClusters; j++) {
				clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointsArr[i].clusterId][0] = 0;
				clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointsArr[i].clusterId][1] = 0;
				clustersSums[(omp_get_thread_num() * totalNumOfClusters) + pointsArr[i].clusterId][2] = 0;
			}
		}
	//clearClusterPoints(clusterArr, clustersPointsArr, totalNumOfPoints, totalNumOfClusters);	//	Clears the clusters' points array pointers
	if (*clustersUnChanged == totalNumOfClusters)
		return FALSE;
	return TRUE;
}

void pointRelocation(point_t* pointsArr, double timeInterval, double currentT, int totalNumOfPoints) {		//	Relocates all points over their circle.
	int halfOfPointsCPU = totalNumOfPoints / 2;
	int halfOfPointsGPU = halfOfPointsCPU;
	point_t* pointsHalfOne, *pointsHalfTwo;
	pointsHalfOne = (point_t*)malloc(sizeof(point_t) * halfOfPointsCPU);	//	Half of points array to be relocated by CPU.
	if (totalNumOfPoints % 2 != 0) 
		halfOfPointsGPU++;
	pointsHalfTwo = (point_t*)malloc(sizeof(point_t) * halfOfPointsGPU);	//	Half of points array to be relocated by GPU.
	memcpy(pointsHalfOne, pointsArr, sizeof(point_t) * halfOfPointsCPU);	//	Copies one half of the points array for the CPU.
	memcpy(pointsHalfTwo, pointsArr + halfOfPointsCPU, sizeof(point_t) * halfOfPointsGPU);	//	Copies the other half of the points array for the GPU.
#pragma omp parallel
		{
#pragma omp single nowait
		pointRelocationCuda(pointsHalfTwo, timeInterval, currentT, halfOfPointsGPU);	//	Sends half of the points held by this node to it's GPU.
	//pointRelocationCuda(pointsArr, timeInterval, currentT, totalNumOfPoints);
		pointRelocationCPU(pointsHalfOne, timeInterval, currentT, halfOfPointsCPU);		//	Calculates the other half using all cores of the CPU.
	//pointRelocationCPU(pointsArr, timeInterval, currentT, totalNumOfPoints);
		}
	memcpy(pointsArr, pointsHalfOne, (sizeof(point_t) * halfOfPointsCPU));		//	Unites the two "sub-arrays" back into the original points array.
	memcpy(pointsArr + halfOfPointsCPU, pointsHalfTwo, sizeof(point_t) * halfOfPointsGPU);
	free(pointsHalfOne);
	free(pointsHalfTwo);
}

void pointRelocationCPU(point_t* points, double timeInterval, double currentT, int numOfPoints) { 		//	Moves all points over their circle, using the CPU's cores.
#pragma omp parallel for
	for (int i = 0; i < numOfPoints; i++) {
		double centerX = points[i].a, centerY = points[i].b;		//	Saves current X, Y coordinates.
		points[i].x = centerX + (points[i].radius * cos((2 * PI * currentT) / timeInterval));		//	Calculates new X coordinate and stores it in the point's X value.
		points[i].y = centerY + (points[i].radius * sin((2 * PI * currentT) / timeInterval));		//	Calculates new Y coordinate and stores it in the point's Y value.
	}
}

void assignToCluster(point_t* pointArr,		//	Determines what cluster the given point belongs to.
	cluster_t* clusterArr, int totalNumOfClusters, int totalNumOfPoints) {
		int i, j;
#pragma omp parallel for private (i, j)
		for(j = 0; j < totalNumOfPoints; j++) {
			int clusterChosen;
			double tempDistance, minDistance = INT_MAX;
			for(i = 0; i < totalNumOfClusters; i++) {		//	Checks all distances from the given point to all clusters.
				tempDistance = calcDistance(pointArr[j].x, pointArr[j].y, clusterArr[i].x, clusterArr[i].y);
				if (tempDistance < minDistance)	{	//	If current distance is smaller than the minimum, change it to be the new minimum distance.
					clusterChosen = clusterArr[i].clusterId;
					minDistance = tempDistance;
					//printf("i = %d, j = %d, clusterchonse = %d, tempDistance = %lf, mindistance = %lf\n", i, j, clusterChosen, tempDistance, minDistance);
				}
			}
			pointArr[j].clusterId = clusterChosen;
			//printf("j = %d, point = %d, cluster chosen = %d\n", j, pointArr[j].id, pointArr[j].clusterId);
		}
}

double calcDistance(double pointAX, double pointAY, double pointBX, double pointBY) {		//	Calculates the distance between two XY points.
	double x = pointAX - pointBX;
	double y = pointAY - pointBY;
	return sqrt(pow(x, 2) + pow(y, 2));
}

/*void clearClusterPoints(cluster_t* cluster, point_t** clustersPointsArr,
	int totalNumOfPoints, int totalNumOfClusters) {		//	Clears the clusters' points array pointers
#pragma omp parallel for
	for(int i = 0; i < totalNumOfClusters; i++) {
		for(int j = 0; j < cluster[i].numOfPoints; j++) {										//	from the clustersPointsArray.
			clustersPointsArr[(totalNumOfPoints * cluster[i].clusterId) + j] = NULL;
		}
		cluster[i].numOfPoints = 0;
	}
}*/

void deterMinClusterDistance(cluster_t *clusterArr, 
	double* distance, int totalNumOfClusters) {	//	Calculates the minimal distance between two clusters.
	double tempDistance, currentMinimum = calcDistance(clusterArr[0].x, clusterArr[0].y, clusterArr[1].x, clusterArr[1].y);
#pragma omp parallel for
	for(int i = 0; i < totalNumOfClusters; i++) {
		for(int j = i+1; j < totalNumOfClusters; j++) {
			tempDistance = calcDistance(clusterArr[i].x, clusterArr[i].y, clusterArr[j].x, clusterArr[j].y);	//	Calculates distance between two current clusters.
			if (tempDistance < currentMinimum)
				currentMinimum = tempDistance;
		}
	}
	*distance = currentMinimum;
}

void createClusters(point_t* pointsArr, cluster_t* clusterArr, int totalNumOfClusters) {//	Creates clusters using the first K points coordinates.
#pragma omp parallel for
	for(int i = 0; i < totalNumOfClusters; i++) {
		clusterArr[i].clusterId = i;
		clusterArr[i].numOfPoints = 0;
		clusterArr[i].x = pointsArr[i].x;
		clusterArr[i].y = pointsArr[i].y;
		clusterArr[i].hasntChanged = FALSE;
	}
}

void saveClustersState(cluster_t* clustersArr, cluster_t* savedClustersArr, int totalNumOfClusters) {		//	Saves clusters state.
	for(int i = 0; i < totalNumOfClusters; i++) {
		savedClustersArr[i].clusterId = clustersArr[i].clusterId;
		savedClustersArr[i].x = clustersArr[i].x;
		savedClustersArr[i].y = clustersArr[i].y;
	}
}

point_t* allocateMainPointsArray(int totalNumOfPoints) {		//	Allocates an array of point_t types for the points being read from the file.
	point_t* pointsArr = (point_t*)malloc(sizeof(point_t) * totalNumOfPoints);
	if (pointsArr == NULL)
		printf("Error allocating memory for the points array.\n");
	return pointsArr;
}

int allocateClusters(double settingsArr[],		//	Allocates memory for the clusters array and the
	cluster_t** clusterArr) {		//	clusters' points array.
	*clusterArr = (cluster_t*)malloc(sizeof(cluster_t) * (int)settingsArr[SET_NUM_OF_CLUSTERS]);
	if (clusterArr == NULL) {
		printf("Error allocating memory for the clusters array.\n");
		return FALSE;
	}
	return TRUE;
}

void masterFinalCode(int totalNumOfClusters, MPI_Comm Comm, cluster_t* savedClustersArr, double minimalDistance, double minimalT,		//	Master's final sequence of code.
	int numprocs, MPI_Status *status, MPI_Datatype clusterDatatype, double** clustersSums, cluster_t* clustersArr, point_t* pointsArr) {
	cluster_t* finalClusterArr = (cluster_t*)malloc(sizeof(cluster_t) * totalNumOfClusters);	//	Final cluster array to be saved to the output file.
		double finalDistance, finalT;		//	Final distance and time T to be saved to the output file.
		finalClusterArr = recvFinalResults(Comm, savedClustersArr, minimalDistance, minimalT, &finalDistance, &finalT, numprocs, totalNumOfClusters, *status, clusterDatatype); // Receives final results from slave nodes.
		writeToFile(finalDistance, finalT, finalClusterArr, totalNumOfClusters);	//	Writes results to the output file.
		freeMemory(clustersSums, clustersArr, pointsArr, totalNumOfClusters);	//	Frees the memory allocated by the master.
}