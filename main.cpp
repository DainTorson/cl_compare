#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

using namespace std;

#define BLOCK_SIZE 16
#define NUM_OF_IDXS 2
#define CHECK(err)                                                       \
    do{                                                                     \
        if (err != CL_SUCCESS){                                          \
            printf("ERROR code %d on line %d\n", err, __LINE__-1);       \
            return -1;                                                   \
        }                                                                \
    } while (0)

const char * loadKernel(char * filename)
{
	string kernel = "";
	ifstream file(filename);
	string line;
	while(getline(file, line))
	{
		kernel += line;
	}

	char * writable = new char[kernel.size() + 1];
	copy(kernel.begin(), kernel.end(), writable);
	writable[kernel.size()] = '\0';

	return writable;
}

bool loadConditions(char * filename, int &startRow, int &endRow, int &startCol, int &endCol)
{
	ifstream file(filename);

	if(file.is_open())
	{
		file >> startRow;
		file >> endRow;
		file >> startCol;
		file >> endCol;

		return true;
	}

	return false;
}

float * loadMatrix(char * filename, int &height, int &width)
{
	ifstream file(filename);
	float * inputData;

	if(file.is_open())
	{
		file >> height;
		file >> width;
		inputData = (float * )malloc(height*width*sizeof(float));

		for(int index = 0; index < height*width && !file.eof(); ++index)
		{
			file >> inputData[index];
		}
	}

	return inputData;
}

template<class T> void printArray(T arr[], int size)
{
	for(int index = 0; index < size; ++index)
	{
		cout << arr[index] << " ";
	}

	cout << endl;
}

template<class T> void printMatrix(T * matrix, int height, int width)
{
	for(int rowIdx = 0; rowIdx < height; ++rowIdx)
	{
		for(int colIdx = 0; colIdx < width; ++colIdx)
		{
			cout << setw(4) << matrix[rowIdx*width + colIdx] << " ";
		}

		cout << endl;
	}
}

void compare(float * hostInput1, float * hostInput2, int * hostOutput, int width,
	int startRow, int endRow, int startCol, int endCol, int inputLength)
{

	*(hostOutput) = inputLength + 1;
	*(hostOutput + 1) = inputLength + 1;

	for(int rowIdx = startRow; rowIdx < endRow; ++rowIdx)
	{
		for(int colIdx = startCol; colIdx < endCol; ++colIdx)
		{
			if(hostInput1[rowIdx*width + colIdx] != hostInput2[rowIdx*width + colIdx])
			{
				*(hostOutput) = rowIdx;
				*(hostOutput + 1) = colIdx;
				return;
			}
		}
	}
}

int parallelCompare(float * hostInput1, float * hostInput2, int * hostOutput, int width,
	int startRow, int endRow, int startCol, int endCol, int inputLength)
{
	cl_int cl_error = CL_SUCCESS;
	cl_platform_id platform;
	cl_error = clGetPlatformIDs(1, &platform, NULL);
	CHECK(cl_error);
	
	cl_device_id device = (cl_device_id)malloc(sizeof(cl_device_id)*1000);
	cl_error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	CHECK(cl_error);
	
	cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &cl_error); 
	CHECK(cl_error);
	
	size_t param_size;
	cl_error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &param_size);
	CHECK(cl_error);
	
	cl_device_id * cl_devices = (cl_device_id*)malloc(param_size);
	cl_error = clGetContextInfo(context, CL_CONTEXT_DEVICES, param_size, cl_devices, NULL);
	CHECK(cl_error);
	
	cl_command_queue cl_cmd_queue = clCreateCommandQueue(context, cl_devices[0], NULL, &cl_error);
	CHECK(cl_error);
	const char * comparesrc = loadKernel("kernel.c");

	cl_program cl_prgm;
	cl_prgm = clCreateProgramWithSource(context, 1, &comparesrc, NULL, &cl_error);
	CHECK(cl_error);
	
	char cl_compile_flags[4096];
	sprintf(cl_compile_flags, "-cl-mad-enable");
	
	cl_error = clBuildProgram(cl_prgm, 0, NULL, cl_compile_flags, NULL, NULL);
	CHECK(cl_error);
	
	cl_kernel kernel = clCreateKernel(cl_prgm, "compare", &cl_error);
	CHECK(cl_error);


	for(int rowIdx = startRow; rowIdx < endRow; ++rowIdx)
	{
		float * hostTempInput1 = (float *)malloc(inputLength*sizeof(float));
		float * hostTempInput2 = (float *)malloc(inputLength*sizeof(float));

		memcpy(hostTempInput1, hostInput1 + rowIdx*width + startCol, inputLength*sizeof(float));
		memcpy(hostTempInput2, hostInput2 + rowIdx*width + startCol, inputLength*sizeof(float));

		*hostOutput = rowIdx;
		*(hostOutput + 1) = inputLength + 1;

		cl_mem deviceInput1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLength * sizeof(float),
								 hostTempInput1, &cl_error);
		CHECK(cl_error);
		cl_mem deviceInput2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLength * sizeof(float),
								 hostTempInput2, &cl_error);
		CHECK(cl_error);

		cl_mem deviceOutput = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), hostOutput + 1, &cl_error);
		CHECK(cl_error);

		size_t global_item_size = ((inputLength - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE;
		size_t local_item_size = BLOCK_SIZE;

		cl_error = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &deviceInput1);
		CHECK(cl_error);
		cl_error = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &deviceInput2);
		CHECK(cl_error);
		cl_error = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &deviceOutput);
		CHECK(cl_error);
		cl_error = clSetKernelArg(kernel, 3, sizeof(int), &inputLength);

		cl_event event = NULL;
		cl_error = clEnqueueNDRangeKernel(cl_cmd_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
		CHECK(cl_error);
	
		cl_error = clWaitForEvents(1, &event);
		CHECK(cl_error);

		clEnqueueReadBuffer(cl_cmd_queue, deviceOutput, CL_FALSE, 0, sizeof(int), hostOutput + 1, 0, NULL, NULL);

		clReleaseMemObject(deviceInput1);
		clReleaseMemObject(deviceInput2);
		clReleaseMemObject(deviceOutput);

		free(hostTempInput1);
		free(hostTempInput2);

		if(*(hostOutput + 1) != inputLength + 1)
		{
			*(hostOutput + 1) += startCol;
			break;
		}
	}

	delete [] comparesrc;
}

int main(void)
{
	float * hostInput1;
	float * hostInput2;
	int * hostOutput;
	int height;
	int width;
	int startRow;
	int endRow;
	int startCol;
	int endCol;
	int inputLength;
	bool parallelExecution = true;

	hostInput1 = loadMatrix("input_data1.txt", height, width);
	hostInput2 = loadMatrix("input_data2.txt", height, width);
	hostOutput = (int *)malloc(NUM_OF_IDXS*sizeof(int));

	printMatrix(hostInput1, height, width);
	cout << endl;
	printMatrix(hostInput2, height, width);
	cout << endl;

	bool success = loadConditions("cond.txt", startRow, endRow, startCol, endCol);
	if(!success)
	{
		return -1;
	}

	if(startRow < 0 || endRow > height || startCol < 0 || endCol > width)
	{
		return -1;
	}

	inputLength = endCol - startCol;

	if(parallelExecution)
	{
		int check = parallelCompare(hostInput1, hostInput2, hostOutput, width,
			startRow, endRow, startCol, endCol, inputLength);
		if(check == -1)
		{
			system("PAUSE"); 
			return -1;
		}
	}
	else
	{
		compare(hostInput1, hostInput2, hostOutput, width, startRow, endRow, startCol, endCol, inputLength);
	}

	if(*(hostOutput + 1) != inputLength + 1)
	{
		printArray(hostOutput, 2);
	}
	else
	{
		cout << "No difference found" << endl;
	}

	free(hostInput1);
	free(hostInput2);
	free(hostOutput);
	
	system("PAUSE"); 
	return 0;
}