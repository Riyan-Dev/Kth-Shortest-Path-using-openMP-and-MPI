#include <mpi.h>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <limits>
#include <fstream>
#include <cstddef>
#include <omp.h>
#include <ctime>



using namespace std;




typedef pair<int, int> pii;


// Function to find K shortest part using dikstra in serial 
void findKShortestSerial( vector<vector<pii> >& g, int n, int m, int k, int src, int destination)
{
 
  
 
    // Vector to store distances
    vector<vector<int> > dis(n + 1, vector<int>(k, 1e9));
 
    // Initialization of priority queue
    priority_queue<pair<int, int>, vector<pair<int, int> >,
                   greater<pair<int, int> > >
        pq;
    pq.push({ 0, src });
    dis[src][0] = 0;
 
    // while pq has elements
    while (!pq.empty()) {
        // Storing the node value
        int u = pq.top().second;
 
        // Storing the distance value
        int d = (pq.top().first);
        pq.pop();
        if (dis[u][k - 1] < d)
            continue;
        vector<pair<int, int> > v = g[u];
 
        // Traversing the adjacency list

        for (int i = 0; i < v.size(); i++) {
            int dest = v[i].first;
            int cost = v[i].second;
 
            // Checking for the cost
            if (d + cost < dis[dest][k - 1]) {
                dis[dest][k - 1] = d + cost;
 
                // Sorting the distances
               
                sort(dis[dest].begin(), dis[dest].end());
 
                // Pushing elements to priority queue
                
                pq.push({ (d + cost), dest });

            }
        }
    }
 
    // Printing K shortest paths
    cout << k << " shortest Path from node " << src << " to destination node " << destination << ": ";
    for (int i = 0; i < k; i++) {
        cout << dis[destination][i] << " ";
    }
    cout << endl;
}

// Function to find K shortest part using dikstra
void findKShortest( vector<vector<pii> >& g, int n, int m, int k, int src, int destination)
{
 
  
 
    // Vector to store distances
    vector<vector<int> > dis(n + 1, vector<int>(k, 1e9));
 
    // Initialization of priority queue
    priority_queue<pair<int, int>, vector<pair<int, int> >,
                   greater<pair<int, int> > >
        pq;
    pq.push({ 0, src });
    dis[src][0] = 0;
 
    // while pq has elements
    while (!pq.empty())
    {
        // Parallelization: Parallelize the outer loop which iterates over elements of priority queue
        #pragma omp parallel for shared(pq, dis) schedule(guided, 2)
        for (int j = 0; j < pq.size(); j++)
        {
            // Storing the node value
            int u, d;
            #pragma omp critical
            {
                u = pq.top().second;
                d = pq.top().first;
                pq.pop();
            }

            // Process neighbors of the current node
            for (int i = 0; i < g[u].size(); i++)
            {
                int dest = g[u][i].first;
                int cost = g[u][i].second;

                // Checking for the cost
                if (d + cost < dis[dest][k - 1] && d + cost >= 0)
                {
                    // Use critical section for updating dis vector
                    #pragma omp critical 
                    {
                        dis[dest][k - 1] = d + cost;
                        sort(dis[dest].begin(), dis[dest].end());
                        // Pushing elements to priority queue
                        pq.push({(d + cost), dest});
                    }
                }
            }
        }
    }
 
    // Printing K shortest paths
    cout << k << " shortest Path from node " << src << " to destination node " << destination << ": ";
    for (int i = 0; i < k; i++) {
        cout << dis[destination][i] << " ";
    }
    cout << endl;
}


// printing grpah for testing purpose
void printGraph(vector<vector<pii> >& graph) {
    for (int u = 0; u < graph.size(); ++u) {
        cout << "Node " << u << " -> ";
        for (int i = 0; i < graph[u].size(); ++i) {
            int v = graph[u][i].first;
            int w = graph[u][i].second;
            cout << "(" << v << ", " << w << ") ";
        }
        cout << endl;
    }
}


// Data Structure to store the edge while rading file
struct Edge{
    int source;
    int dest;
    int weight;
    Edge(): source(0), dest(0), weight(0) {}
    Edge(int s, int d, int w){
        source = s;
        dest = d;
        weight = w;
    }
    void print(){
        cout << source << "     " << dest << "     " << weight << endl;
    }
};


// Function to create mpi data type for edge structure
MPI_Datatype create_mpi_edge_type() {
    const int nitems = 3;  // Total number of fields
    int blocklengths[3] = {1, 1, 1};  // Number of elements in each block
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};  // MPI data types of each field
    MPI_Datatype mpi_edge_type;
    MPI_Aint offsets[3];

    //Setting offsets for each variable of Edge objedct
    offsets[0] = offsetof(Edge, source);
    offsets[1] = offsetof(Edge, dest);
    offsets[2] = offsetof(Edge, weight);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_edge_type);
    MPI_Type_commit(&mpi_edge_type);

    return mpi_edge_type;
}

// FUnction for String to int to read the value of k
int StrToInt(char* str){

    int num = 0;
    for(int i = 0; str[i]; i++){
        num = num*10 + (str[i]-'0');
    }
    return num;

}


// Function serial Execution 
void serial(vector<Edge>& edges, int k, int* testingNodes){

    // Start the timer
    clock_t start = clock();

    cout << "-----------------------------------Serial Execution of 10 pairs-------------------------------------" << endl;
    vector<vector<pii> > graph;

    int m = edges.size();

    // resing the graph vector of vector according to the number of our nodes
    graph.resize(edges[m-1].source + 1);

    for (int i = 0; i < m; i++){
            graph[edges[i].source].push_back({edges[i].dest, edges[i].weight});
            // graph[edges[i][0]].push_back({edges[i][1], edges[i][2]});

            // edges[i].print();     
    } 
    for (int i = 0, j = 0 ; i < 10; i++){
        int src = testingNodes[j++];
        int dest = testingNodes[j++];

    // cout << src << " " << dest << endl;

        findKShortestSerial(graph, edges[m-1].source, m, k, src, dest);
    }

    
    // End the timer
    clock_t end = clock();

    // Calculate the duration
    double duration = double(end - start) / CLOCKS_PER_SEC;

    // Output the duration
     std::cout << "Serial Execution time: " << duration << " seconds" << std::endl;

}


// Main Function 
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    srand(time(0) + rank);

    // Access command-line arguments using argv
    if (argc < 3){
        cout << "Value of K and Num oF threads Missing" << endl;
        return 1;
    }
    int k = StrToInt(argv[1]);
    int num_threads = StrToInt(argv[2]);
    omp_set_num_threads(num_threads);
    int m = 0; // To store number of edges
    vector<Edge> edges; // To store edges
    int nodes = 0; // To store number of Nodes
    vector<vector<pii> > graph; // To store the graph

    int testingNodes[20] = {0}; // for testing nodes in serial
    int testingNodesProcess[2]= {0}; // for testing nodes in each process during parralel execution
    clock_t start;
    clock_t end;
    

    // vector of vector of pair {dest, weight}

    // 1 -> {2, 3}, {3, 4}
    // 2 -> 

    // file reading and serial execution in master node
    if (rank == 0) {
       ifstream file("Email-Enron.txt");

        if (!file.is_open()) {
            cerr << "Error opening file." << endl;
            return 1;
        }
        int currentSource = 0;
        
       // reading file in parallel using openMP 
        // #pragma omp parallel default(none) shared(file, edges, nodes, m, currentSource)
    {
        int source, dest;
        // #pragma omp for reduction(+:nodes, m)
        for (; file >> source >> dest;) {
            ++m;
            if (currentSource != source) {
                ++nodes;
                currentSource = source;
            }
            edges.push_back(Edge(source, dest, 1));
        }
    }
        // printing number of edges and nodes
        cout << edges[m-1].source << "  " << m << endl;
        // int nodes = 4; 
        // int m = 6;
        // int edges[][3]  = { { 1, 2, 1 }, { 1, 3, 3 }, { 2, 3, 2 },
        //     { 2, 4, 6 }, { 3, 2, 8 }, { 3, 4, 1 } };

        // graph.resize(nodes + 1);
        
        // Getting random 10 pair of nodes
        for (int i = 0, j = 0; i < 10; i++){
            testingNodes[j++] = rand()%edges[m-1].source;
            testingNodes[j++] = rand()%edges[m-1].source;

            
        }

        // serial Executrion 
       serial(edges, k, testingNodes);

        // Startingf clock for parralel execution
        start = clock();

        cout << endl << endl << "-----------------------------------Parralel Execution of 10 pairs-------------------------------------" << endl; 

    }

    // Scattering tetsingNodes accross the  working nodes
    MPI_Scatter(testingNodes, 2, MPI_INT, testingNodesProcess, 2, MPI_INT, 0, MPI_COMM_WORLD);

    //getting number of vertexes
    int vertexSize = edges.size();


    //broad Casting number of vertex in other processes.
    MPI_Bcast(&vertexSize, 1, MPI_INT, 0,  MPI_COMM_WORLD);


    // resizing vector of edges in working nodes
    if (rank != 0){
        edges.resize(vertexSize);
    }

    // getting edge datatype nfor MPI
    MPI_Datatype mpi_edge_type = create_mpi_edge_type();

    // broadcasting edges across the working nodes 
    MPI_Bcast(edges.data(), vertexSize, mpi_edge_type, 0,MPI_COMM_WORLD);
    m = vertexSize;

    graph.resize(edges[m-1].source + 1);

    // making adjency list / graph in parralal using openMP 
    // #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < m; i++){
            // #pragma omp critical 
            graph[edges[i].source].push_back({edges[i].dest, edges[i].weight});
            // graph[edges[i][0]].push_back({edges[i][1], edges[i][2]});

            // edges[i].print();
        
            
    } 
    
    // printGraph(graph);
    // getting source and destination nodes
    int src = testingNodesProcess[0];
    int dest = testingNodesProcess[1];

    // cout << src << " " << dest << endl;

    // Running the algorithm in parallel
    findKShortest(graph, edges[m-1].source, m, k, src, dest);

    MPI_Type_free(&mpi_edge_type);

    // waiting for all working nodes to finish execution
    MPI_Barrier(MPI_COMM_WORLD);
    

    // getting execution time in master node 
    if (rank == 0){
        end = clock();

        // Calculate the duration
        double duration = double(end - start) / CLOCKS_PER_SEC;

        // Output the duration
        std::cout << "Parralel Execution time: " << duration << " seconds" << std::endl;
    }
    

    

   

    MPI_Finalize();
    return 0;
}

