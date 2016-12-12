#include "main_header.h"

//	Reads information from the input file.
int readFile(double settingsArr[], point_t** pointsArr, cluster_t** clustersArr) {	
	char* inputFile = INPUT_FILE_LOCATION;
	int firstLineReadStatus, readSuccess = TRUE;
	FILE* file = fopen(inputFile, "r");	//	Opens the input file.
	if (file == NULL) {		//	If file didn't open right, or file doesn't exist.
		printf("Error opening the input file.\n");
		fflush(stdout);
		readSuccess = FALSE;
	}
	firstLineReadStatus = fscanf(file, "%lf %lf %lf %lf %lf\n", &settingsArr[SET_NUM_OF_POINTS], //	Reads main information (number of points, number of clusters, delta-t,
		&settingsArr[SET_NUM_OF_CLUSTERS], &settingsArr[SET_DELTA_T],			//	total time T, and the maximum number of iterations for the alogrithm).
		&settingsArr[SET_TIME_INTERVAL], &settingsArr[SET_ITERATIONS_LIMIT]);	
																				
	if (firstLineReadStatus != NUM_OF_FIRST_PARAM) {	//	If not enough parameters were read, or there was a matching error.
		printf("Error reading first line from the input file.\n");
		readSuccess = FALSE;
	}
	*pointsArr = allocateMainPointsArray((int)settingsArr[SET_NUM_OF_POINTS]);		//	Allocates an array of point_t types for the points being read from the file.
	if (*pointsArr == NULL) 
		readSuccess = FALSE;
	readSuccess = pointReadLoop(*pointsArr, (int)settingsArr[SET_NUM_OF_POINTS], file);		//	Reads all points.
	if (!readSuccess) 
		free(*pointsArr);		//	If read did not succeed, function will free the points array and return FALSE to the calcMeans func.
	fclose(file);		//	Closes the file.
	return readSuccess;
}

//	Reads the points from the input file.
int pointReadLoop(point_t* pointsArr, int numOfPoints, FILE* file) {
	int pointsReadStatus, pointId, readSuccess = TRUE;
	double pointCenterX, pointCenterY, pointRadius;
	for(int i = 0; i < numOfPoints; i++) {
		pointsReadStatus = fscanf(file, "%d %lf %lf %lf\n", &pointId, &pointCenterX, &pointCenterY, &pointRadius);
		if (pointsReadStatus != NUM_OF_POINT_PARAM) {		//	If number of arguments read from file does not match 5.
			printf("Error reading points from the input file.\n");
			readSuccess = FALSE;
		}
		(pointsArr)[i].id = pointId;
		(pointsArr)[i].a = pointCenterX;
		(pointsArr)[i].b = pointCenterY;
		(pointsArr)[i].radius = pointRadius;
	}
	return readSuccess;
}

//	Writes the results to the output file.
int writeToFile(double distance, double T, cluster_t* savedClustersArr, int totalNumOfClusters) {
	char* outputFile = OUTPUT_FILE_LOCATION;
	FILE* file = fopen(outputFile, "w");
	if (file == NULL) {			//	If opening the output file did not succeed, function will return FALSE.
		printf("Error writing to output file, program will now exit.\n");
		fclose(file);
		return FALSE;
	}
	fprintf(file, "d = %lf\n", distance);		//	Prints the minimal distance found.
	fprintf(file, "t = %lf\n", T);				//	Prints the time in which the minimal distance was found.
	fprintf(file, "Centers of the clusters:\n");
	for(int i = 0; i < totalNumOfClusters; i++)		//	Prints the centers of the clusters in the minimal distance T.
		fprintf(file, "Id = %d, X = %lf, Y = %lf\n", savedClustersArr[i].clusterId, savedClustersArr[i].x, savedClustersArr[i].y);
	fclose(file);
	return TRUE;
}