# Nihal Ranchod - 2427378

import math

def euclidean_distance(x,u):
    return math.sqrt((x[0] - u[0])**2 + (x[1] - u[1])**2)

def find_closest_cluster(data_point, cluster_centres):
    min_distance = float('inf')
    closest_centroid = None
    for cluster_centre in cluster_centres:
        distance = euclidean_distance(data_point, cluster_centre)
        if distance < min_distance:
            min_distance = distance
            closest_centroid = cluster_centre
    
    return closest_centroid

def update_cluster_centres(clusters):
    new_cluster_centres = []
    
    for cluster in clusters:
        x_sum = 0
        y_sum = 0
        if len(cluster) == 0:
            continue
        for data_point in cluster:
            x_sum += data_point[0]
            y_sum += data_point[1]
        
        new_cluster_centre = (x_sum / len(cluster), y_sum / len(cluster))
        new_cluster_centres.append(new_cluster_centre)

    return new_cluster_centres

def k_means(dataset, initial_cluster_centres):
    cluster_centres = initial_cluster_centres
    previous_centres = None
    while previous_centres != cluster_centres:
        clusters = [[] for _ in range(len(cluster_centres))]
        for data_point in dataset:
            closest_cluster_centre = find_closest_cluster(data_point, cluster_centres)
            cluster_index = cluster_centres.index(closest_cluster_centre)
            clusters[cluster_index].append(data_point)

        previous_centres = cluster_centres[:]
        cluster_centres = update_cluster_centres(clusters)

    return cluster_centres

def sum_of_squares_error(dataset, cluster_centres):
    error = 0
    for data_point in dataset:
        closest_cluster_centre = find_closest_cluster(data_point, cluster_centres)
        error += euclidean_distance(data_point, closest_cluster_centre) ** 2
    
    return round(error, 4)

def main():
    dataset = [
    (0.22, 0.33), (0.45, 0.76), (0.73, 0.39), (0.25, 0.35), (0.51, 0.69),
    (0.69, 0.42), (0.41, 0.49), (0.15, 0.29), (0.81, 0.32), (0.50, 0.88),
    (0.23, 0.31), (0.77, 0.30), (0.56, 0.75), (0.11, 0.38), (0.81, 0.33),
    (0.59, 0.77), (0.10, 0.89), (0.55, 0.09), (0.75, 0.35), (0.44, 0.55)
    ]
    
    initial_cluster_centres = []

    for i in range(3):
        x = float(input())
        y = float(input())
        cluster_centre = (x,y)
        initial_cluster_centres.append(cluster_centre)

    #print(f'Initial Cluster Centres: {initial_cluster_centres}')

    final_cluster_centres = k_means(dataset, initial_cluster_centres)
    #print(f'Final cluster centres: {final_cluster_centres}')
    error = sum_of_squares_error(dataset, final_cluster_centres)
    #print(f'Sum of sqaures error: {error}')
    print(error)
    

if '__main__' == __name__:
    main()