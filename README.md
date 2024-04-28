Description:


**Setting up the Environment**

**Installing WSL/Ubuntu**

To run MPI program we need to first install WSL using the command:

	wsl --install
 
Wait for the installation of ubuntu. 


**Create a new directory**
In home folder in wsl create a new directory using command

  mkdir PDC
  
**Clone Github repo**
Clone  MPICH environment to setup the cluster for running mpi program 

  git clone https://github.com/NLKNguyen/alpine-mpich
  
**Install Dockers**

Using the link provided below 

https://docs.docker.com/desktop/install/ubuntu/

**Compiling the MPI Program**

To compile the MPI program, follow these steps:
Navigate to the directory containing the Docker file (dockerfile).
add the following command to the docker file to compile the program:
  RUN mpic++ -o proj Project.cpp -std=c++11 -fopenmp
To compile the program on master run first navigate to the director:
  user/PDC/alpine-mpich/cluster
Run the following command to compile the program on master and add / run the worker in the cluster environment:
  ./cluster.sh up size=4
This will create a cluster of 1 master and 3 workers/slave machines
This command will compile the Project.cpp file using c++ 11 and openMP support  and generate an executable named proj.

**Running the Executable**

After compiling the program, you can run the executable using the mpirun command. Here's how:
To compile the program on master run first navigate to the director:
  user/PDC/alpine-mpich/cluster
Ensure that you have MPI installed and configured on your system.
Use the following command to run the executable:
		./clsuter.sh exec mpirun -np 10 ./proj 8 2

In this command:

-np 10 specifies the number of processes to run, in this case, 10 processes.
./proj is the path to the compiled executable file.
8 represents the value of k in the kth shortest path problem.
2 represents the value of num of threads to be executed in openMP

This command will execute the MPI program with 10 processes, finding the 
8 shortest paths for 10 random pairs of starting and destination nodes.

