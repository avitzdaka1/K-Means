# KMeans
K-Means clustering algorithm implemented in C using MPI, OpenMP and Cuda.

This program will calculate cluster means from a given set of points.

At first, the program will read the first line from points.txt which represents the program's configuration arguments:

First number = Number of points,
Second number = Number of K clusters,
Third number = Delta T,
Fourth number = Total T,
Fifth number = Maximum iterations.

The following lines contain randomly generated points:

First number = Point id,
Second number = X value of the point's circle center,
Third number = Y value of the point's circle center,
Fourth number = Radius of the point's circle.

Program can be run with either 1 computer (with a Nvidia GPU), or multiple computers (using wmpiexec.exe).

Only the master node knows the location of the file.

The program in general:

1. The master reads the file configuration and the points.
2. The master sends all points to all other nodes.
3. Each node (including the master) calculates a different t (time) from 0 to total T.
4. Every new T, all points are being relocated on their respective circle (this is achieved using Cuda).
5. After the points have been relocated, each point is assigned to the closest cluster (this is achieved using OpenMP).
6. After all points have been assigned to clusters, each cluster is being re-calculated using the average of all its assigned points (this is achieved using OpenMP).
7. Instructions 5-6 repeat untill all clusters have not changed or X (iterations limit) iterations have been made.
8. Program determines the minimal distance between any two clusters, and saves the result.
9. Instructions 4-8 repeat untill t is greater than total T.
10. Master node gathers results from all other nodes, compares them with his result, and finds the minimal distance between any two clusters.
11. Master writes the result to the output file and prints the time (in seconds) that the program ran.

Thoughts about complexity:
At first, I used both Cuda and OpenMP to calculate the pointRelocation function, but then I found out that it doesn't work well, so now I only use Cuda in this task.
The same goes for the other functions, for example the points assignment function, it feels like using Cuda will only slow the program.

Please send me an email to omermyaari@gmail.com for any questions.
