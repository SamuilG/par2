//
// Starting code for the MPI coursework.
//
// See lectures and/or the worksheet corresponding to this part of the module for instructions
// on how to build and launch MPI programs. A simple makefile has also been included (usage optional).
//


//
// Includes.
//

// Standard includes.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// The MPI library.
#include <mpi.h>

// Some extra routines for this coursework. DO NOT MODIFY OR REPLACE THESE ROUTINES,
// as this file will be replaced with a different version for assessment.
#include "cwk2_extra.h"


//
// Main.
//
int main( int argc, char **argv )
{
    int i;

    //
    // Initialisation.
    //

    // Initialise MPI and get the rank of this process, and the total number of processes.
    int rank, numProcs;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &numProcs );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank     );

    // Check that the number of processes is a power of 2, but <=256, so the data set, which is a multiple of 256 in length,
    // is also a multiple of the number of processes. If using OpenMPI, you may need to add the argument '--oversubscribe'
    // when launching the executable, to allow more processes than you have cores.
    if( (numProcs&(numProcs-1))!=0 || numProcs>256 )
    {
        // Only display the error message from one processes, but finalise and quit all of them.
        if( rank==0 ) printf( "ERROR: Launch with a number of processes that is a power of 2 (i.e. 2, 4, 8, ...) and <=256.\n" );

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Load the full data set onto rank 0.
    float *globalData = NULL;
    int globalSize = 0;
    if( rank==0 )
    {
        globalData = readDataFromFile( &globalSize );           // globalData must be free'd on rank 0 before quitting.
        if( globalData==NULL )
        {
            MPI_Finalize();                                     // Should really communicate to all other processes that they need to quit as well ...
            return EXIT_FAILURE;
        }

        printf( "Rank 0: Read in data set with %d floats.\n", globalSize );
    }

    // Calculate the number of floats per process. Note that only rank 0 has the correct value of localSize
    // at this point in the code. This will somehow need to be communicated to all other processes. Note also
    // that we can assume that globalSize is a multiple of numProcs.
    int localSize = globalSize / numProcs;          // = 0 at this point of the code for all processes except rank 0.

    // Start the timing now, after the data has been loaded (will only output on rank 0).
    double startTime = MPI_Wtime();


    //
    // Task 1: Calculate the mean using all available processes.
    //
    float mean = 0.0f;          // Your calculated mean should be placed in this variable.


    // broadcast to all processes
    MPI_Bcast(&globalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // calc floats per process
    float *localData = (float *)malloc(localSize * sizeof(float));
    if (localData == NULL)
    {


        printf("Error allocating local data on rank %d.\n", rank);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // scatter global data from rank 0 to all of the processes
    MPI_Scatter(globalData, localSize, MPI_FLOAT,
                localData, localSize, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // ecery process: calculates the sum of its local data
    float localSum = 0.0f;
    for (i = 0; i < localSize; i++)
    {

        localSum += localData[i];
    }

    //localSum values
    int partner;   // partner process - pp
    float received_Sum; // received sum

    for (int step = 1; step < numProcs; step *= 2)
    {
        if (rank % (2 * step) == 0)
        {
            partner = rank + step;
            if (partner < numProcs)
            {



                MPI_Recv(&received_Sum, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                localSum += received_Sum;

            }
        }
        else
        {
            partner = rank - step;

            MPI_Send(&localSum, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD);
            break; // if sent
        }
    }

    if (rank == 0)
    {
        mean = localSum / globalSize;
    }


    //
    // Task 2. Calculate the variance using all processes.
    //
    float variance = 0.0f;      // Your calculated variance should be placed in this variable.
    
    
    // binary tree broadcast; every process gets the mean
    for (int step = numProcs / 2; step >= 1; step /= 2)
    {
        if (rank < step)
        {
            int sender = rank + step;


            if (sender < numProcs)
            {
                MPI_Recv(&mean, 1, MPI_FLOAT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        else if (rank < 2 * step)
        {
            int receiver = rank - step;

            MPI_Send(&mean, 1, MPI_FLOAT, receiver, 1, MPI_COMM_WORLD);
        }
    }
    
    

    // eveery process calculates the sum of squared differences for its local data
    float localVarianceSum = 0.0f;
    // calculate the variance sum
    for (i = 0; i < localSize; i++)
    {

        float diff = localData[i] - mean;
        localVarianceSum += diff * diff;
    }




    // reduce the variance sum using a binary tree reduction
    for (int step = 1; step < numProcs; step *= 2)
    {
        if (rank % (2 * step) == 0)
        {
            partner = rank + step;
            if (partner < numProcs)
            {
                
                MPI_Recv(&received_Sum, 1, MPI_FLOAT, partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                localVarianceSum += received_Sum;


            }
        }
        else
        {
            partner = rank - step;


            MPI_Send(&localVarianceSum, 1, MPI_FLOAT, partner, 2, MPI_COMM_WORLD);
            break;
        }
    }

    if ( rank == 0 )
    {
        
        variance = localVarianceSum / globalSize;
        
    }
    free(localData);

    //
    // Output the results alongside a serial check.
    //
    if( rank==0 )
    {
        // Output the results of the timing now, before moving onto other calculations.
        printf( "Total time taken: %g s\n", MPI_Wtime() - startTime );

        // Your code MUST call this function after the mean and variance have been calculated using your parallel algorithms.
        // Do not modify the function itself (which is defined in 'cwk2_extra.h'), as it will be replaced with a different
        // version for the purpose of assessing. Also, don't just put the values from serial calculations here or you will lose marks.
        finalMeanAndVariance( mean, variance );

        // Check the answers against the serial calculations. This also demonstrates how to perform the calculations
        // in serial, which you may find useful. Note that small differences in floating point calculations between
        // equivalent parallel and serial codes are possible, as explained in Lecture 11.

        // Mean.
        float sum = 0.0;
        for( i=0; i<globalSize; i++ ) sum += globalData[i];
        float mean = sum / globalSize;

        // Variance.
        float sumSqrd = 0.0;
        for( i=0; i<globalSize; i++ ) sumSqrd += ( globalData[i]-mean )*( globalData[i]-mean );
        float variance = sumSqrd / globalSize;

        printf( "SERIAL CHECK: Mean=%g and Variance=%g.\n", mean, variance );

   }

    //
    // Free all resources (including any memory you have dynamically allocated), then quit.
    //
    if( rank==0 ) free( globalData );

    MPI_Finalize();

    return EXIT_SUCCESS;
}
