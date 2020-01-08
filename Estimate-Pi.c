/////////////////////////////////////////////////////////////////
//
// Kenta Suzue
//
// Estimate-Pi.c
//
// Compile:  mpicc -g -Wall Estimate-Pi.c -o Estimate-Pi
//
// Run:      mpiexec -n <p> ./Estimate-Pi <N>
//
//           <p>: the number of processes
//           <N>: the number of dart tosses
//
///////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <mpi.h>

void Error_Check(int argc, char** argv, long long int* Num_Darts, MPI_Comm mpi_comm, int comm_sz, int my_rank);
long long int Darts_In_Circle(long long int number_of_tosses, int my_rank);
double Random_Double(int max, int min);

int main(int argc, char** argv)
{
    int comm_sz, my_rank;

    long long int num_darts;
    long long int num_darts_per_process;
    long long int number_in_circle;
    long long int num_darts_in_circle_global;
    double pi_estimate;

    //Seed the random number generator to get different results each run
    //Differently seeded random number generator useful for dart throws at random x and y coordinates of dartboard
    srand(time(NULL));

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Process 0 reads in the total number of tosses into the variable num_darts and broadcasts it to the other processes
    Error_Check(argc, argv, &num_darts, MPI_COMM_WORLD, comm_sz, my_rank);

    num_darts_per_process = num_darts / ((long long int) comm_sz);

    // printf("1The number of dart throws is %lld from rank %d!\n", num_darts, my_rank);    

    number_in_circle = Darts_In_Circle(num_darts_per_process, my_rank);

    // printf("2The number of dart throws is %lld from rank %d!\n", num_darts, my_rank);    

    // printf("The number of dart throws per process is %lld from rank %d!\n", num_darts_per_process, my_rank);    

    // printf("The number of darts thrown in the circle is %lld from rank %d!\n", number_in_circle, my_rank);    

    //Find the global sum of the local variable number_in_circle
    MPI_Reduce(&number_in_circle, &num_darts_in_circle_global, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);    

    // if (my_rank == 0)
    //    printf("The total number of darts thrown in the circle is %lld from rank %d!\n", num_darts_in_circle_global, my_rank);    

    // Process 0 prints the result of MPI_Reduce to find the global sum of the local variable number_in_circle
    if (my_rank == 0)
        pi_estimate = 4.0 * ((double) num_darts_in_circle_global)/((double) num_darts);

    // if (my_rank == 0)
    //     printf("pi estimate is %f, from process %d\n", pi_estimate, my_rank);

    if (my_rank == 0)
    {
        printf("Pi is estimated to be %f, based on the Monte Carlo method with %lld dart tosses.\n", pi_estimate, num_darts);
        printf("The result combined the result of %d processes, each process analyzing %lld dart tosses.\n", comm_sz, num_darts_per_process);
    }

    MPI_Finalize();    

    return 0;
}

//The function Error_Check uses process 0 to read in the total number of dart throws from the command line argument.
//The function Error_Check also uses process 0 to perform error checking on the command line arguments.
//The number of arguments must be 2.
//The total number of dart throws must be: positive, within the range of a long long int, 
//and evenly divisible by the number of processes 
void Error_Check(int argc, char** argv, long long int* num_darts, MPI_Comm mpi_comm, int comm_sz, int my_rank)
{
    if (my_rank == 0)
    {
        //Error checking that the number of arguments is 2. If not, print help for format of run command.
        if (argc != 2)
        {
            fprintf(stderr, "Error: The number of arguments is incorrect!\n");
            fprintf(stderr, "USAGE: mpiexec -n <number_of_processes> ./Estimate-Pi <number_of_dart_throws>\n");

            *num_darts = -1LL;            
        }

        else
        {
            //Process 0 reads in the total number of dart throws.
            //Store the argument for the number of dart throws into the long long int variable num_darts. 
            //with the atoll() function that converts a character string to a long long integer.  

            *num_darts = atoll(argv[1]);

            //Error checking that the number of dart throws is a positive integer. If not, print help that the the number of dart throws should be a positive integer.
            if (*num_darts < 1LL)
            {
                fprintf(stderr, "Error: The number of dart throws should be a positive integer!\n");
                *num_darts = -1LL;            
            }

            else
            {
                //find the number of digits in LLONG_MAX, such that the string num_darts_str will be long enough
                long long int llong_max_quotient = LLONG_MAX;
                int llong_max_digit_count = 0;

                do {
                    llong_max_quotient /= 10LL;
                    llong_max_digit_count++;
                } while (llong_max_quotient > 0LL);

                // printf("Number of digits in LLONG_MAX is %d\n", llong_max_digit_count);

                //Error checking that the number of dart throws is in the range of a long long int. If not, print help with the positive range between 1 and LLONG_MAX (maximum value for a long long int).
                //Error checking compares (i) the string length of the argument for the number of dart throws and
                //with (ii) the number of digits in LLONG_MAX.
                if (strlen(argv[1]) >  llong_max_digit_count)
                {
                    printf("strlen is %zu and exceeds %d\n", strlen(argv[1]), llong_max_digit_count);
                    fprintf(stderr, "Error: The number of dart throws should be a positive integer in the range between 1 and %lld!\n", LLONG_MAX);
                    *num_darts = -1LL;
                }                 

                else
                {
                    char num_darts_str[llong_max_digit_count + 1];
                    sprintf(num_darts_str, "%lld", *num_darts);
                    //printf("The string num_darts_str is %s", num_darts_str);

                    // printf("Num_Darts is %lld\n", *num_darts);

                    //Error checking that the number of dart throws is in the range of a long long int. If not, print help with the positive range between 1 and LLONG_MAX (maximum value for a long long int).
                    //Error checking converts the string to a long long int, and back from a long long int to a string.
                    //If the original string isn't recovered, then the original string did not represent a valid long long long int.
                    if (strcmp(argv[1], num_darts_str) != 0)
                    {
                        fprintf(stderr, "Error: The number of dart throws should be a positive integer in the range between 1 and %lld!\n", LLONG_MAX);
                        *num_darts = -1LL;
                    }
	
                    //Error checking that the number of dart throws is evenly divisible by the number of processes
                    else if (*num_darts % comm_sz != 0LL)   
                    {
                        fprintf(stderr, "Error: The number of dart throws should be evenly divisible by the number of processes!\n");
                        *num_darts = -1LL;
                    }
                }
            }
        }
    }

    //Process 0 broadcasts num_darts to all of the processes
    MPI_Bcast(num_darts, 1, MPI_LONG_LONG, 0, mpi_comm);

    if (*num_darts < 0LL)
    {
        MPI_Finalize();
        exit (-1);
    }
}

// The function Darts_In_Circle returns the number of darts that hit inside a 1 foot radius circle.
// The parameters are the number of darts thrown and the number of the process calling the function
long long int Darts_In_Circle(long long int number_of_tosses, int my_rank)
{
    long long int number_in_circle = 0; 
    long long int toss;

    for (toss = 0L; toss < number_of_tosses; toss++) { 

        double x = Random_Double(1, -1);
        double y = Random_Double(1, -1);
        // printf("random x is %f, random y is %f from process %d\n", x, y, my_rank);

        double distance_squared = x*x + y*y; 

        // printf("distance squared is %f from process %d\n", distance_squared, my_rank);

        if (distance_squared <= 1.0)
             number_in_circle++; 
    }

    return number_in_circle; 
}

// The function Random_Double returns a random double that is in the range between the parameters max and min.
double Random_Double(int max, int min) {

    //random double between 0.0 and 1.0
    double random_0_to_1 = ((double)rand()) / (double)RAND_MAX;
    // printf("random_0_to_1 is %f\n", random_0_to_1);

    //random double between 0 and (max - min)
    double random_0_to_max_minus_min = random_0_to_1 * (max - min);
    // printf("random_0_to_1 is %f\n", random_0_to_max_minus_min);

    //random double between min and max
    return random_0_to_max_minus_min + (double) min;
}
