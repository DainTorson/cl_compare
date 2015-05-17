__kernel void compare(__global float * input1, __global float * input2, __global int * output,
	int inputLength)
{
	int id = get_global_id(0);

	if(id < inputLength)
	{
		if(input1[id] != input2[id])
		{
			atomic_min(output, id);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
}