#pragma once
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <omp.h>

#define PI 3.14159265358979323846	//	Constant for PI.
#define TRUE 1	
#define FALSE 0
#define NUM_OF_FIRST_PARAM 5		//	Number of parameters for a successful read from the input file's first line.
#define NUM_OF_POINT_PARAM 4		//	Number of parameters for a successful point read from the input file.
#define MAX_FILE_NAME 20			//	Maximum length for a file name.
#define MASTER_PROCESS 0			//	Master process number.
#define NUM_OF_POINT_FIELDS 7		//	Number of fields in the point struct.
#define NUM_OF_CLUSTER_FIELDS 7		//	Number of fields in the cluster struct.
#define NUM_OF_INITIAL_SETTINGS 5	//	Number of inital settings (from the input file's first line).
#define MAX_CUDA_THREADS 1024		//	Maximum number of Cuda threads the program will use.
#define SET_NUM_OF_POINTS 0			//	Value for the number of points in the initial settings array.
#define SET_NUM_OF_CLUSTERS 1		//	Value for the number of clusters in the initial settings array.
#define SET_DELTA_T 2				//	Value for delta t in the initial settings array.
#define SET_TIME_INTERVAL 3			//	Value for the time interval in the initial settings array.
#define SET_ITERATIONS_LIMIT 4		//	Value for the iterations limit in the initial settings array.
#define CLUSTER_SUM_X 0
#define CLUSTER_SUM_Y 1
#define CLUSTER_SUM_NUM_OF_POINTS 2
#define INPUT_FILE_LOCATION "J:\\Google Drive\\Software Engineering Afeka\\2016 Summer\\Parallel Distributed Computing\\Parallel_Final_PRoject_Coherent\\Parallel_Final_Project\\points.txt"
#define OUTPUT_FILE_LOCATION "J:\\Google Drive\\Software Engineering Afeka\\2016 Summer\\Parallel Distributed Computing\\Parallel_Final_PRoject_Coherent\\Parallel_Final_Project\\points_output.txt"

//	X,Y Point struct.
typedef struct point {		
	int id, clusterId;
	double a, b, x, y, radius;
} point_t;

//	Cluster struct.
typedef struct cluster {	
	int clusterId, numOfPoints, hasntChanged;
	double x, y, oldX, oldY;
} cluster_t;

//	Used by master to send the initial settings, points and clusters to all slaves.
void sendSettings(MPI_Comm Comm, point_t* pointsArr, double settingsArr[], MPI_Datatype pointDatatype, MPI_Datatype clusterDatatype);

//	Used by slaves to receive the initial settings, points and clusters.
int recvSettings(MPI_Comm Comm, point_t** pointsArr, double settingsArr[], MPI_Datatype pointDatatype, MPI_Datatype clusterDatatype);

//	Creates MPI datatype for the cluster struct.
MPI_Datatype createClusterDatatype();		

//	Creates MPI datatype for the point struct.
MPI_Datatype createPointDatatype();		

//	Frees allocated memory.
void freeMemory(double** clustersSums, cluster_t* clustersArr, point_t* pointsArr, int totalNumOfClusters);	

//	Master's final sequence of code.
void masterFinalCode(int totalNumOfClusters, MPI_Comm Comm, cluster_t* savedClustersArr, double minimalDistance, double minimalT,		
	int numprocs, MPI_Status *status, MPI_Datatype clusterDatatype, double** clustersSums, cluster_t* clustersArr, point_t* pointsArr);

//	Allocates an array of point_t types for the points being read from the file.
point_t* allocateMainPointsArray(int totalNumOfPoints);		

//	Allocates memory for the clusters array.
int allocateClusters(double settingsArr[], cluster_t** clusterArr);

//	Point relocation function using GPU.
cudaError_t pointRelocationCuda(point_t* pointsArr, double timeInterval, double currentT, int numOfPoints, double cosT, double sinT);	

//	Creates clusters using the first K points coordinates.
void createClusters(point_t* pointsArr, cluster_t* clusterArr, int totalNumOfClusters); 

//	Sends final results to the master.
void sendFinalResults(MPI_Comm comm, cluster_t* slaveClustersArr, double slaveDistance, int numOfClusters, double slaveT, MPI_Datatype clusterDataType);	

//	Receives final results from all slave nodes.
cluster_t* recvFinalResults(MPI_Comm comm, cluster_t* masterClustersArr, double masterDistance, double masterT,	
	double* resultDistance, double* resultT, int numprocs, int numOfClusters, MPI_Status status, MPI_Datatype clusterDatatype);

//	Saves clusters state.
void saveClustersState(cluster_t* clustersArr, cluster_t* savedClustersArr, int totalNumOfClusters);

//	Reads information from the input file.
int readFile(double settingsArr[], point_t** pointsArr, cluster_t** clustersArr);	

//	Reads the points from the input file.
int pointReadLoop(point_t* pointsArr, int numOfPoints, FILE* file);

//	Writes the results to the output file.
int writeToFile(double distance, double T, cluster_t* savedClustersArr, int totalNumOfClusters);	

//	Main job code.
int calcKMeans(MPI_Comm Comm, MPI_Status *status, int numprocs, MPI_Datatype pointDatatype, MPI_Datatype clusterDatatype, int myMpiID);

//	Code sequence of the actual algorithm calculation.
void TCalc(int myMpiID, double settingsArr[], int numprocs, point_t* pointsArr, cluster_t* clustersArr,		
	double* minimalDistance, double* minimalT, cluster_t* savedClustersArr, double** clustersSums);

//	Moves all points over their circle.
void pointRelocation(point_t* points , double timeInterval, double currentT, int totalNumOfPoints);		

//	Determines what cluster the given point belongs to.
void assignToCluster(point_t* pointArr,	cluster_t* clusterArr, int totalNumOfClusters, int totalNumOfPoints, double** clustersSums);

//	Calculates the distance between two XY points.
double calcDistance(double pointAX, double pointAY, double pointBX, double pointBY);	

//	Calculates new center for each cluster.
int calcNewClustCenter(double** clustersSums, cluster_t* clusterArr, point_t* pointsArr, int totalNumOfClusters, int totalNumOfPoints, int* clustersChanged);

//	Clears the clusters' points array pointers
void clearClusterPoints(cluster_t* cluster, point_t** clustersPointsArr, int totalNumOfPoints, int totalNumOfClusters);

//	Calculates the minimal distance between two clusters.
void deterMinClusterDistance(cluster_t *clusterArr, double* distance, int totalNumOfClusters);

//	Allocates memory space for the points array in the GPU and copies the points array to that memory.
cudaError_t allocatePointsCuda(point_t* pointsArr, int numOfPoints);