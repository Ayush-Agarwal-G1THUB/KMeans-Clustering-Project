/*
* K-Means Clustering Implementation for Iris Dataset
* Author: Ayush Agarwal [Ayush-Agarwal-G1THUB]
* Date: 31/05/2025
* Description: This program implements the K-Means clustering algorithm
* to classify the Iris dataset. It includes functions for
* data loading, centroid initialization, assignment,
* updating, running multiple iterations, and testing accuracy.
*/

#include<iostream>
#include<string>
#include <vector>
#include <cmath>
#include <limits>
#include <ctime>
#include <random>
#include <algorithm>
#include <map>

using namespace std;

struct labelled_x {
// Represents a data point with four features
// and the species label
double feature1; // Sepal length
double feature2; // Sepal width
double feature3; // Petal length
double feature4; // Petal width
string species_label; // known species label
};

// The Iris dataset, containing measurements for different Iris species
// Each entry includes four features and a species label
vector<labelled_x> dataset = {
{5.1,3.5,1.4,0.2,"Iris-setosa"},
{4.9,3.0,1.4,0.2,"Iris-setosa"},
{4.7,3.2,1.3,0.2,"Iris-setosa"},
{4.6,3.1,1.5,0.2,"Iris-setosa"},
{5.0,3.6,1.4,0.2,"Iris-setosa"},
{5.4,3.9,1.7,0.4,"Iris-setosa"},
{4.6,3.4,1.4,0.3,"Iris-setosa"},
{5.0,3.4,1.5,0.2,"Iris-setosa"},
{4.4,2.9,1.4,0.2,"Iris-setosa"},
{4.9,3.1,1.5,0.1,"Iris-setosa"},
{5.4,3.7,1.5,0.2,"Iris-setosa"},
{4.8,3.4,1.6,0.2,"Iris-setosa"},
{4.8,3.0,1.4,0.1,"Iris-setosa"},
{4.3,3.0,1.1,0.1,"Iris-setosa"},
{5.8,4.0,1.2,0.2,"Iris-setosa"},
{5.7,4.4,1.5,0.4,"Iris-setosa"},
{5.4,3.9,1.3,0.4,"Iris-setosa"},
{5.1,3.5,1.4,0.3,"Iris-setosa"},
{5.7,3.8,1.7,0.3,"Iris-setosa"},
{5.1,3.8,1.5,0.3,"Iris-setosa"},
{5.4,3.4,1.7,0.2,"Iris-setosa"},
{5.1,3.7,1.5,0.4,"Iris-setosa"},
{4.6,3.6,1.0,0.2,"Iris-setosa"},
{5.1,3.3,1.7,0.5,"Iris-setosa"},
{4.8,3.4,1.9,0.2,"Iris-setosa"},
{5.0,3.0,1.6,0.2,"Iris-setosa"},
{5.0,3.4,1.6,0.4,"Iris-setosa"},
{5.2,3.5,1.5,0.2,"Iris-setosa"},
{5.2,3.4,1.4,0.2,"Iris-setosa"},
{4.7,3.2,1.6,0.2,"Iris-setosa"},
{4.8,3.1,1.6,0.2,"Iris-setosa"},
{5.4,3.4,1.5,0.4,"Iris-setosa"},
{5.2,4.1,1.5,0.1,"Iris-setosa"},
{5.5,4.2,1.4,0.2,"Iris-setosa"},
{4.9,3.1,1.5,0.1,"Iris-setosa"},
{5.0,3.2,1.2,0.2,"Iris-setosa"},
{5.5,3.5,1.3,0.2,"Iris-setosa"},
{4.9,3.1,1.5,0.1,"Iris-setosa"},
{4.4,3.0,1.3,0.2,"Iris-setosa"},
{5.1,3.4,1.5,0.2,"Iris-setosa"},
{5.0,3.5,1.3,0.3,"Iris-setosa"},
{4.5,2.3,1.3,0.3,"Iris-setosa"},
{4.4,3.2,1.3,0.2,"Iris-setosa"},
{5.0,3.5,1.6,0.6,"Iris-setosa"},
{5.1,3.8,1.9,0.4,"Iris-setosa"},
{4.8,3.0,1.4,0.3,"Iris-setosa"},
{5.1,3.8,1.6,0.2,"Iris-setosa"},
{4.6,3.2,1.4,0.2,"Iris-setosa"},
{5.3,3.7,1.5,0.2,"Iris-setosa"},
{5.0,3.3,1.4,0.2,"Iris-setosa"},
{7.0,3.2,4.7,1.4,"Iris-versicolor"},
{6.4,3.2,4.5,1.5,"Iris-versicolor"},
{6.9,3.1,4.9,1.5,"Iris-versicolor"},
{5.5,2.3,4.0,1.3,"Iris-versicolor"},
{6.5,2.8,4.6,1.5,"Iris-versicolor"},
{5.7,2.8,4.5,1.3,"Iris-versicolor"},
{6.3,3.3,4.7,1.6,"Iris-versicolor"},
{4.9,2.4,3.3,1.0,"Iris-versicolor"},
{6.6,2.9,4.6,1.3,"Iris-versicolor"},
{5.2,2.7,3.9,1.4,"Iris-versicolor"},
{5.0,2.0,3.5,1.0,"Iris-versicolor"},
{5.9,3.0,4.2,1.5,"Iris-versicolor"},
{6.0,2.2,4.0,1.0,"Iris-versicolor"},
{6.1,2.9,4.7,1.4,"Iris-versicolor"},
{5.6,2.9,3.6,1.3,"Iris-versicolor"},
{6.7,3.1,4.4,1.4,"Iris-versicolor"},
{5.6,3.0,4.5,1.5,"Iris-versicolor"},
{5.8,2.7,4.1,1.0,"Iris-versicolor"},
{6.2,2.2,4.5,1.5,"Iris-versicolor"},
{5.6,2.5,3.9,1.1,"Iris-versicolor"},
{5.9,3.2,4.8,1.8,"Iris-versicolor"},
{6.1,2.8,4.0,1.3,"Iris-versicolor"},
{6.3,2.5,4.9,1.5,"Iris-versicolor"},
{6.1,2.8,4.7,1.2,"Iris-versicolor"},
{6.4,2.9,4.3,1.3,"Iris-versicolor"},
{6.6,3.0,4.4,1.4,"Iris-versicolor"},
{6.8,2.8,4.8,1.4,"Iris-versicolor"},
{6.7,3.0,5.0,1.7,"Iris-versicolor"},
{6.0,2.9,4.5,1.5,"Iris-versicolor"},
{5.7,2.6,3.5,1.0,"Iris-versicolor"},
{5.5,2.4,3.8,1.1,"Iris-versicolor"},
{5.5,2.4,3.7,1.0,"Iris-versicolor"},
{5.8,2.7,3.9,1.2,"Iris-versicolor"},
{6.0,2.7,5.1,1.6,"Iris-versicolor"},
{5.4,3.0,4.5,1.5,"Iris-versicolor"},
{6.0,3.4,4.5,1.6,"Iris-versicolor"},
{6.7,3.1,4.7,1.5,"Iris-versicolor"},
{6.3,2.3,4.4,1.3,"Iris-versicolor"},
{5.6,3.0,4.1,1.3,"Iris-versicolor"},
{5.5,2.5,4.0,1.3,"Iris-versicolor"},
{5.5,2.6,4.4,1.2,"Iris-versicolor"},
{6.1,3.0,4.6,1.4,"Iris-versicolor"},
{5.8,2.6,4.0,1.2,"Iris-versicolor"},
{5.0,2.3,3.3,1.0,"Iris-versicolor"},
{5.6,2.7,4.2,1.3,"Iris-versicolor"},
{5.7,3.0,4.2,1.2,"Iris-versicolor"},
{5.7,2.9,4.2,1.3,"Iris-versicolor"},
{6.2,2.9,4.3,1.3,"Iris-versicolor"},
{5.1,2.5,3.0,1.1,"Iris-versicolor"},
{5.7,2.8,4.1,1.3,"Iris-versicolor"},
{6.3,3.3,6.0,2.5,"Iris-virginica"},
{5.8,2.7,5.1,1.9,"Iris-virginica"},
{7.1,3.0,5.9,2.1,"Iris-virginica"},
{6.3,2.9,5.6,1.8,"Iris-virginica"},
{6.5,3.0,5.8,2.2,"Iris-virginica"},
{7.6,3.0,6.6,2.1,"Iris-virginica"},
{4.9,2.5,4.5,1.7,"Iris-virginica"},
{7.3,2.9,6.3,1.8,"Iris-virginica"},
{6.7,2.5,5.8,1.8,"Iris-virginica"},
{7.2,3.6,6.1,2.5,"Iris-virginica"},
{6.5,3.2,5.1,2.0,"Iris-virginica"},
{6.4,2.7,5.3,1.9,"Iris-virginica"},
{6.8,3.0,5.5,2.1,"Iris-virginica"},
{5.7,2.5,5.0,2.0,"Iris-virginica"},
{5.8,2.8,5.1,2.4,"Iris-virginica"},
{6.4,3.2,5.3,2.3,"Iris-virginica"},
{6.5,3.0,5.5,1.8,"Iris-virginica"},
{7.7,3.8,6.7,2.2,"Iris-virginica"},
{7.7,2.6,6.9,2.3,"Iris-virginica"},
{6.0,2.2,5.0,1.5,"Iris-virginica"},
{6.9,3.2,5.7,2.3,"Iris-virginica"},
{5.6,2.8,4.9,2.0,"Iris-virginica"},
{7.7,2.8,6.7,2.0,"Iris-virginica"},
{6.3,2.7,4.9,1.8,"Iris-virginica"},
{6.7,3.3,5.7,2.1,"Iris-virginica"},
{7.2,3.2,6.0,1.8,"Iris-virginica"},
{6.2,2.8,4.8,1.8,"Iris-virginica"},
{6.1,3.0,4.9,1.8,"Iris-virginica"},
{6.4,2.8,5.6,2.1,"Iris-virginica"},
{7.2,3.0,5.8,1.6,"Iris-virginica"},
{7.4,2.8,6.1,1.9,"Iris-virginica"},
{7.9,3.8,6.4,2.0,"Iris-virginica"},
{6.4,2.8,5.6,2.2,"Iris-virginica"},
{6.3,2.8,5.1,1.5,"Iris-virginica"},
{6.1,2.6,5.6,1.4,"Iris-virginica"},
{7.7,3.0,6.1,2.3,"Iris-virginica"},
{6.3,3.4,5.6,2.4,"Iris-virginica"},
{6.4,3.1,5.5,1.8,"Iris-virginica"},
{6.0,3.0,4.8,1.8,"Iris-virginica"},
{6.9,3.1,5.4,2.1,"Iris-virginica"},
{6.7,3.1,5.6,2.4,"Iris-virginica"},
{6.9,3.1,5.1,2.3,"Iris-virginica"},
{5.8,2.7,5.1,1.9,"Iris-virginica"},
{6.8,3.2,5.9,2.3,"Iris-virginica"},
{6.7,3.3,5.7,2.5,"Iris-virginica"},
{6.7,3.0,5.2,2.3,"Iris-virginica"},
{6.3,2.5,5.0,1.9,"Iris-virginica"},
{6.5,3.0,5.2,2.0,"Iris-virginica"},
{6.2,3.4,5.4,2.3,"Iris-virginica"},
{5.9,3.0,5.1,1.8,"Iris-virginica"}
};

/**
* @brief Loads the dataset into training and test sets.
* @param training_set Output: Vector to store training data.
* @param test_set Output: Vector to store test data.
* @param test_set_size The desired number of samples to include in the test set.
* The remaining data will be used for training.
*/
void loadSets (vector<labelled_x> &training_set, vector<labelled_x> &test_set, const int& test_set_size) {
// shuffle the dataset
unsigned seed = chrono::system_clock::now().time_since_epoch().count();
default_random_engine rng(seed);
vector<labelled_x> shuffledDataset = dataset;
shuffle(shuffledDataset.begin(), shuffledDataset.end(), rng);

for (int i = 0; i < shuffledDataset.size(); i++) {
// put n items in test_set
if (i % (shuffledDataset.size() / test_set_size) == 0) {
test_set.push_back(
{
shuffledDataset[i].feature1,
shuffledDataset[i].feature2,
shuffledDataset[i].feature3,
shuffledDataset[i].feature4,
shuffledDataset[i].species_label
}
);
}

// put all others in training_set
else {
training_set.push_back(
{
shuffledDataset[i].feature1,
shuffledDataset[i].feature2,
shuffledDataset[i].feature3,
shuffledDataset[i].feature4,
shuffledDataset[i].species_label
}
);
}
}
}

/**
* @brief Initialises each centroid by picking unique random points from the training set.
* @param centroids Output: Vector to store the initialised centroids.
* @param num_centroids Number of centroids to initialise.
* @param training_set The dataset used to pick initial centroid locations.
*/
void initialiseCentroids (vector<labelled_x> &centroids, const int &num_centroids, vector<labelled_x> &training_set) {
centroids.resize(num_centroids);
for (labelled_x &centroid : centroids) {
centroid.feature1 = training_set[rand()%training_set.size()].feature1;
centroid.feature2 = training_set[rand()%training_set.size()].feature2;
centroid.feature3 = training_set[rand()%training_set.size()].feature3;
centroid.feature4 = training_set[rand()%training_set.size()].feature4;
}
}

/**
* @brief Computes the squared Euclidean distance (cost) between a centroid and a data point.
* @param centroid The centroid point.
* @param point The data point.
* @return The squared Euclidean distance between the centroid and the point.
*/
double computeCost (const labelled_x &centroid, const labelled_x &point) {
return pow(centroid.feature1 - point.feature1, 2) +
pow(centroid.feature2 - point.feature2, 2) +
pow(centroid.feature3 - point.feature3, 2) +
pow(centroid.feature4 - point.feature4, 2);
}

/**
* @brief Assigns each training data point to its closest centroid.
* @param c_assigned Output: Vector where each element at index i stores the index of the centroid
* to which training_set[i] is assigned.
* @param training_set The dataset to assign to centroids.
* @param centroids The current centroids.
*/
void assignClosestCentroids (vector<int>& c_assigned, const vector<labelled_x> &training_set, const vector<labelled_x> &centroids) {
c_assigned.resize(training_set.size());

for (int i = 0; i<training_set.size(); i++) {
labelled_x point = training_set[i];
double mincost = numeric_limits<double>::max();

for (int j = 0; j<centroids.size(); j++) {
labelled_x centroid = centroids[j];
double cost = computeCost(centroid, point);
if (cost < mincost) {
mincost = cost;
c_assigned[i] = j;
}
}
}
}

/**
* @brief Updates the position of each centroid based on the mean of its assigned data points.
* Handles empty clusters by reinitializing the centroid to a random training point.
* @param centroids Output: Vector of centroids, whose positions will be updated.
* @param num_centroids The total number of centroids.
* @param training_set The training data points.
* @param c_assigned The current assignments of data points to centroids.
*/
void updateCentroids (vector<labelled_x> &centroids, const int &num_centroids, vector<labelled_x> &training_set, vector<int> &c_assigned) {
vector<labelled_x> centroidAverage(num_centroids, {0.0, 0.0, 0.0, 0.0});
vector<int> countAssigned(num_centroids, 0);

for (int i = 0; i<training_set.size(); i++) {
// For every point in the training set
// Add its values to the corresponding centroid position
// in the centroidAverage vector
centroidAverage[c_assigned[i]].feature1 += training_set[i].feature1;
centroidAverage[c_assigned[i]].feature2 += training_set[i].feature2;
centroidAverage[c_assigned[i]].feature3 += training_set[i].feature3;
centroidAverage[c_assigned[i]].feature4 += training_set[i].feature4;

countAssigned[c_assigned[i]]++;
}

// divide by total number of points assigned to that centroid
for (int i = 0; i<centroidAverage.size(); i++) {
if (countAssigned[i] == 0) {
centroids[i] = training_set[rand() % training_set.size()];
}
else {
centroids[i] = {
centroidAverage[i].feature1 / countAssigned[i],
centroidAverage[i].feature2 / countAssigned[i],
centroidAverage[i].feature3 / countAssigned[i],
centroidAverage[i].feature4 / countAssigned[i]
};
}
}
}

/**
* @brief Executes the K-Means clustering algorithm for a specified number of iterations.
* This involves iteratively assigning points to centroids and updating centroid positions.
* @param centroids Output: The final positions of the centroids after clustering.
* @param num_centroids The number of centroids.
* @param training_set The dataset used for clustering.
* @param c_assigned Output: The final assignments of training points to centroids.
* @param num_iters The number of iterations to run the clustering algorithm.
*/
void clusteringAlgorithm (vector<labelled_x> &centroids, const int &num_centroids, vector<labelled_x> &training_set, vector<int> &c_assigned, const int num_iters) {
for (int i = 0; i<num_iters; i++) {
assignClosestCentroids(c_assigned, training_set, centroids);
updateCentroids(centroids, num_centroids, training_set, c_assigned);
}
}

/**
* @brief Runs the K-Means clustering algorithm multiple times with different initializations
* and selects the set of centroids that results in the minimum total cost.
* @param num_runs The number of times to run the K-Means algorithm.
* @param centroids Output: The best set of centroids found across all runs.
* @param num_centroids The number of centroids.
* @param training_set The training data used for clustering.
* @param c_assigned Output: The assignments corresponding to the best set of centroids.
* @param num_iters The number of iterations for each K-Means run.
*/
void runs (const int &num_runs, vector<labelled_x> &centroids, const int &num_centroids, vector<labelled_x> &training_set, vector<int> &c_assigned, const int num_iters) {
double minCost = numeric_limits<double>::max();
vector<labelled_x> minCentroids(num_centroids);
vector<int> bestCAssigned(training_set.size());
for (int i = 0; i<num_runs; i++) {
initialiseCentroids(centroids, num_centroids, training_set);
clusteringAlgorithm(centroids, num_centroids, training_set, c_assigned, num_iters);

double totalCost = 0;
for (int j = 0; j<training_set.size(); j++) {
labelled_x point = training_set[j];
totalCost += computeCost(centroids[c_assigned[j]], point);
}

if (totalCost < minCost) {
minCost = totalCost;
minCentroids = centroids;
bestCAssigned = c_assigned;
}
}

centroids = minCentroids;
c_assigned = bestCAssigned;

cout << "Centroids :" << endl;
for (labelled_x centroid : centroids) {
cout << centroid.feature1 << ", "
<< centroid.feature2 << ", "
<< centroid.feature3 << ", "
<< centroid.feature4 << endl;
}
}

/**
* @brief Assigns a species label to each centroid based on the majority species of its assigned training points.
* @param centroidLabels Output: Vector where element at index i stores the species label for centroid i.
* @param c_assigned The assignments of training points to centroids.
* @param centroids The final centroids (used for size).
* @param training_set The training data with original species labels.
*/
void assignLabelToCentroids (vector<string> &centroidLabels, vector<int> &c_assigned, vector<labelled_x> &centroids, vector<labelled_x> &training_set) {
for (int centroidIdx = 0; centroidIdx<centroids.size(); centroidIdx++) {
vector<int> cluster;
for (int i = 0; i<c_assigned.size(); i++) {
if (c_assigned[i] == centroidIdx) {
cluster.push_back(i);
}
}

map<string, int> m;
for (int i = 0; i<cluster.size(); i++) {
labelled_x point = training_set[cluster[i]];
m[point.species_label]++;
}

int max_occurrence = 0;
for (auto it : m) {
if (it.second > max_occurrence) {
max_occurrence = it.second;
centroidLabels[centroidIdx] = it.first;
}
}
}
}

/**
* @brief Tests the accuracy of the K-Means clustering by classifying test data points
* and comparing predicted labels with actual labels.
* @param centroids The final trained centroids.
* @param test_set The unseen data points for testing.
* @param centroidLabels The assigned species labels for each centroid.
*/
void test (vector<labelled_x> &centroids, vector<labelled_x> &test_set, vector<string> &centroidLabels) {
vector<int> predictedCentroidIdx(test_set.size());
int correct = 0;

for (int t = 0; t < test_set.size(); t++) {
auto test = test_set[t];
double minCost = numeric_limits<double>::max();

for (int i = 0; i<centroids.size(); i++) {
labelled_x centroid = centroids[i];

double cost = computeCost(centroid, test);
if (cost < minCost) {
minCost = cost;
predictedCentroidIdx[t] = i;
}
}

cout << "\nTest value " << test.feature1 << ", "
<< test.feature2 << ", "
<< test.feature3 << ", "
<< test.feature4 << endl;
cout << "Predicted Label = " << centroidLabels[predictedCentroidIdx[t]] << endl;
cout << " Actual Label = " << test.species_label << endl;

if (test.species_label == centroidLabels[predictedCentroidIdx[t]]) {
cout << "----- CORRECT -----" << endl;
correct++;
}
else cout << "----- WRONG -----" << endl;
}

cout << "\n--- K-Means Test Results ---" << endl;
cout << "Total test samples : " << test_set.size() << endl;
cout << "Correctly classified: " << correct << endl;
double accuracy = 100*correct/test_set.size();
cout << "Accuracy = " << accuracy << "%" << endl;
}

int main()
{
srand(time(0));

// Define parameters for the K-Means algorithm
const int test_set_size = 5; // Number of samples to reserve for the test set
const int num_centroids = 3; // K value: Number of clusters (expected Iris species)
const int num_iters = 100; // Number of iterations for each K-Means run (convergence steps)
const int num_runs = 20; // Number of times to run K-Means with different initializations
// to find the best clustering (to avoid local minima)

// Vectors to hold the data and clustering results
vector<labelled_x> training_set;
vector<labelled_x> test_set;
vector<labelled_x> centroids;
vector<int> c_assigned; // Stores the centroid index assigned to each training point
vector<string> centroidLabels(num_centroids); // Stores the derived species label for each centroid

// Step 1 : Prepare the dataset
loadSets(training_set, test_set, test_set_size);
// Step 2 : perform clustering algorithm multiple times to find the best result
runs(num_runs, centroids, num_centroids, training_set, c_assigned, num_iters);
// Step 3 : Assign the plurality label of each cluster to its centroid
assignLabelToCentroids(centroidLabels, c_assigned, centroids, training_set);
// Step 4 : Evaluate the model's performance on unseen data
test(centroids, test_set, centroidLabels);

return 0;
}