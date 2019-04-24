#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#pragma comment(lib, "glew32.lib")
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <chrono>

using namespace std;

#define CSC(call) {							\
    cudaError err = call;						\
    if(err != cudaSuccess) {						\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));		\
        exit(1);							\
    }									\
} while (0)

#define square(x) ((x)*(x))
#define THREADS 128

struct particle
{
	double2 coord;
	double2 velocity;
	double2 best_coord;
	double2 repultion_force;
};


const int width = 1280;
const int height = 720;

const int particle_cnt = 4000;
const double inertia = 0.981;
const double coef_local = 0.4;
const double coef_global = 0.15;
const double coef_repultion = 0.5;
const double dt = 0.07;

double scale_x = 500;
double scale_y = scale_x * height / width;

const dim3 blocks2D(128, 128);
const dim3 threads2D(32, 32);
const int threads_reduce = 1024;
const int blocks_reduce = width * height / threads_reduce + 1;
const int threads1D = THREADS;
const int blocks1D = ceil((double)particle_cnt / THREADS);

__constant__ double pi = 3.1415;
__constant__ int seed = 1234;

__device__ double dev_center_x = 0;
__device__ double dev_center_y = 0;
__device__ double dev_func_min;
__device__ double dev_func_max;
__device__ double image[height * width];
__device__ double2 g_best;

curandState* dev_states;

struct cudaGraphicsResource *res;
particle *dev_swarm;


double *arr_max_after_reduce_dev;
double *arr_min_after_reduce_dev;

double2 *global_best_after_reduce;

GLuint vbo;


__device__ double rosenbrock(double2 arg) {
	return square((1 - arg.x)) + 100 * square((arg.y - square(arg.x)));
}


__device__ double rosenbrock(int i, int j, double scale_x, double scale_y) {
	double x = 2.0f * i / (double)(width - 1) - 1.0f;
	double y = 2.0f * j / (double)(height - 1) - 1.0f;
	return rosenbrock(make_double2(x * scale_x + dev_center_x, -y * scale_y + dev_center_y));
}


__global__ void rosenbrock_image(double scale_x, double scale_y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for (int j = idy; j < height; j += offsety)
	{
		for (int i = idx; i < width; i += offsetx)
		{
			image[j * width + i] = rosenbrock(i, j, scale_x, scale_y);
		}
	}
}


__global__ void minmax_reduce(double *arr_min_after_reduce, double *arr_max_after_reduce)
{
	__shared__ double shared_min[threads_reduce];
	__shared__ double shared_max[threads_reduce];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < width * height)
	{
		shared_min[threadIdx.x] = image[idx];
		shared_max[threadIdx.x] = image[idx];
	}
	else
	{
		shared_min[threadIdx.x] = INFINITY;
		shared_max[threadIdx.x] = -INFINITY;
	}
	__syncthreads();

	for (int step = 2; step <= threads_reduce; step *= 2)
	{
		if (threadIdx.x * (step + 1) - 1 < threads_reduce)
		{
			shared_min[threadIdx.x * (step + 1) - 1] = (shared_min[threadIdx.x * (step + 1) - 1] < shared_min[threadIdx.x * (step + 1) - step / 2 - 1]) ? shared_min[threadIdx.x * (step + 1) - 1] : shared_min[threadIdx.x * (step + 1) - step / 2 - 1];
			shared_max[threadIdx.x * (step + 1) - 1] = (shared_max[threadIdx.x * (step + 1) - 1] > shared_max[threadIdx.x * (step + 1) - step / 2 - 1]) ? shared_max[threadIdx.x * (step + 1) - 1] : shared_max[threadIdx.x * (step + 1) - step / 2 - 1];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		arr_min_after_reduce[blockIdx.x] = shared_min[threads_reduce - 1];
		arr_max_after_reduce[blockIdx.x] = shared_max[threads_reduce - 1];
	}

}

__global__ void minmax(double *arr_min_after_reduce, double *arr_max_after_reduce, int size)
{
	double min = arr_min_after_reduce[0];
	double max = arr_max_after_reduce[0];
	for (int i = 1; i < size; i++)
	{
		if (arr_min_after_reduce[i] < arr_min_after_reduce[i - 1])
			min = arr_min_after_reduce[i];
		if (arr_max_after_reduce[i] > arr_max_after_reduce[i - 1])
			max = arr_max_after_reduce[i];
	}
	dev_func_min = min;
	dev_func_max = max;
}

__device__ uchar4 get_color(double f) {
	float k = 1.0 / 6.0;
	if (f <= 0)
		return make_uchar4(0, 0, 0, 0);
	if (f < k)
		return make_uchar4((int)(f * 255 / k), 0, 0, 0);
	if (f < 2 * k)
		return make_uchar4(255, (int)((f - k) * 255 / k), 0, 0);
	if (f < 3 * k)
		return make_uchar4(255, 255, (int)((f - 2 * k) * 255 / k), 0);
	if (f < 4 * k)
		return make_uchar4(255 - (int)((f - 3 * k) * 255 / k), 255, 255, 0);
	if (f < 5 * k)
		return make_uchar4(0, 255 - (int)((f - 4 * k) * 255 / k), 255, 0);
	if (f <= 6 * k)
		return make_uchar4(0, 0, 255 - (int)((f - 5 * k) * 255 / k), 0);
	return make_uchar4(0, 0, 0, 0);
}

__global__ void heatmap(uchar4* data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for (int j = idy; j < height; j += offsety)
	{
		for (int i = idx; i < width; i += offsetx)
		{
			data[j * width + i] = get_color((image[j * width + i] - dev_func_min) / (dev_func_max - dev_func_min));
		}
	}
}


__global__ void update_coords_and_velocities(double inertia, double coef_local, double coef_global, double dt, double coef_repultion,
	particle *swarm, int particle_cnt, uchar4* data, double scale_x, double scale_y, curandState * state)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int offsetx = blockDim.x * gridDim.x;

	for (int i = idx; i < particle_cnt; i += offsetx)
	{
		swarm[idx].velocity.x = inertia * swarm[idx].velocity.x + (coef_local * curand_uniform(&state[idx]) * (swarm[idx].best_coord.x - swarm[idx].coord.x) +
			coef_global * curand_uniform(state) * (g_best.x - swarm[idx].coord.x) + coef_repultion * swarm[idx].repultion_force.x) * dt;
		swarm[idx].velocity.y = inertia * swarm[idx].velocity.y + (coef_local * curand_uniform(&state[idx]) * (swarm[idx].best_coord.y - swarm[idx].coord.y) +
			coef_global * curand_uniform(state) * (g_best.y - swarm[idx].coord.y) + coef_repultion * swarm[idx].repultion_force.y) * dt;
		swarm[idx].coord.x += swarm[idx].velocity.x * dt;
		swarm[idx].coord.y += swarm[idx].velocity.y * dt;
		if (rosenbrock(make_double2(swarm[idx].coord.x, swarm[idx].coord.y)) < rosenbrock(make_double2(swarm[idx].best_coord.x, swarm[idx].best_coord.y)))
		{
			swarm[idx].best_coord.x = swarm[idx].coord.x;
			swarm[idx].best_coord.y = swarm[idx].coord.y;
		}
		double2 particle_draw_coord;
		particle_draw_coord.x = (((swarm[idx].coord.x - dev_center_x) / scale_x) + 1) * (width - 1) / 2;
		particle_draw_coord.y = (1 - ((swarm[idx].coord.y - dev_center_y) / scale_y)) * (height - 1) / 2;

		if (particle_draw_coord.x > 0 && particle_draw_coord.x < width && particle_draw_coord.y > 0 && particle_draw_coord.y < height)
		{
			data[(int)particle_draw_coord.y * width + (int)particle_draw_coord.x] = make_uchar4(255, 255, 255, 255);
		}
	}
}

__global__ void repulsive_force(particle *swarm, int particle_cnt)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int offsetx = blockDim.x * gridDim.x;
	double square_dist;

	for (int i = idx; i < particle_cnt; i += offsetx)
	{
		for (int i = 0; i < particle_cnt; i++)
		{
			square_dist = square(swarm[i].coord.x - swarm[idx].coord.x) + square(swarm[i].coord.y - swarm[idx].coord.y);
			swarm[idx].repultion_force.x -= (swarm[i].coord.x - swarm[idx].coord.x) / (square(square_dist) + 1e-3);
			swarm[idx].repultion_force.y -= (swarm[i].coord.y - swarm[idx].coord.y) / (square(square_dist) + 1e-3);
		}
	}
}

__global__ void update_window_center(particle *swarm, int particle_cnt)
{
	double2 sum;
	for (int i = 0; i < particle_cnt; i++)
	{
		sum.x += swarm[i].coord.x;
		sum.y += swarm[i].coord.y;
	}
	sum.x /= particle_cnt;
	sum.y /= particle_cnt;
	dev_center_x = sum.x;
	dev_center_y = sum.y;
}


__global__ void global_best_reduce(particle *swarm, double2 *global_best_after_reduce)
{
	__shared__ double2 shared_min[threads_reduce];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	shared_min[threadIdx.x] = swarm[idx].coord;
	__syncthreads();

	for (int step = 2; step <= threads_reduce; step *= 2)
	{
		if (threadIdx.x * (step + 1) - 1 < threads_reduce)
		{
			shared_min[threadIdx.x * (step + 1) - 1] = (rosenbrock(shared_min[threadIdx.x * (step + 1) - 1]) < rosenbrock(shared_min[threadIdx.x * (step + 1) - step / 2 - 1])) ?
				shared_min[threadIdx.x * (step + 1) - 1] : shared_min[threadIdx.x * (step + 1) - step / 2 - 1];
		}
		__syncthreads();
		step *= 2;
	}
	if (threadIdx.x == 0)
	{
		global_best_after_reduce[blockIdx.x] = shared_min[threads_reduce - 1];
	}
}

__global__ void global_best_final(particle *swarm, double2 *global_best_after_reduce, int size)
{
	double2 max;
	if (size > 0)
	{
		max = global_best_after_reduce[0];
	}
	else
	{
		max = swarm[0].coord;
	}
	for (int i = 1; i < size; i++)
	{
		if (rosenbrock(global_best_after_reduce[i]) < rosenbrock(global_best_after_reduce[i - 1]))
			max = global_best_after_reduce[i];
	}
	for (int i = THREADS * size; i < size; i++)
	{
		if (rosenbrock(swarm[i].coord) < rosenbrock(max))
			max = swarm[i].coord;
	}
	if (rosenbrock(max) < rosenbrock(g_best))
		g_best = max;
}


//__global__ void global_best_final(particle *swarm, double2 *global_best_after_reduce, int size, int particle_cnt)
//{
//		for (int i = 0; i < size; i++)
//		{
//			if (rosenbrock(global_best_after_reduce[i]) < rosenbrock(g_best))
//				g_best = global_best_after_reduce[i];
//		}
//		for (int i = size; i < size; i++)
//		{
//			if (rosenbrock(swarm[i].coord) < rosenbrock(g_best))
//				g_best = swarm[i].coord;
//		}
//}

__global__ void swarm_start(particle *swarm, int particle_cnt, curandState * state)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int offsetx = blockDim.x * gridDim.x;

	for (int i = idx; i < particle_cnt; i += offsetx)
	{
		curand_init(seed, idx, 0, &state[idx]);

		swarm[idx].best_coord.x = swarm[idx].coord.x = curand_uniform(&state[idx]) * width * cos((double)idx / THREADS * 2 * pi);
		swarm[idx].best_coord.y = swarm[idx].coord.y = curand_uniform(&state[idx]) * height * sin((double)idx / THREADS * 2 * pi);

		swarm[idx].velocity = make_double2(0, 0);
		swarm[idx].repultion_force = make_double2(0, 0);

	}
}


void update() {
	uchar4* heat_image;
	size_t size;
	CSC(cudaGraphicsMapResources(1, &res, 0));
	CSC(cudaGraphicsResourceGetMappedPointer((void**)&heat_image, &size, res));

	float time;
	cudaEvent_t start, stop;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start, 0));

	update_window_center << <1, 1 >> > (dev_swarm, particle_cnt);
	CSC(cudaGetLastError());

	rosenbrock_image << <blocks2D, threads2D >> > (scale_x, scale_y);
	CSC(cudaGetLastError());

	minmax_reduce << <blocks_reduce, threads_reduce >> > (arr_min_after_reduce_dev, arr_max_after_reduce_dev);
	CSC(cudaGetLastError());

	minmax << <1, 1 >> > (arr_min_after_reduce_dev, arr_max_after_reduce_dev, blocks_reduce);
	CSC(cudaGetLastError());

	heatmap << <blocks2D, threads2D >> > (heat_image);
	CSC(cudaGetLastError());

	repulsive_force << <blocks1D, threads1D >> > (dev_swarm, particle_cnt);
	CSC(cudaGetLastError());

	update_coords_and_velocities << <blocks1D, threads1D >> > (inertia, coef_local, coef_global, dt, coef_repultion, dev_swarm, particle_cnt, heat_image, scale_x, scale_y, dev_states);
	CSC(cudaGetLastError());

	global_best_reduce << <blocks_reduce, threads_reduce >> > (dev_swarm, global_best_after_reduce);
	CSC(cudaGetLastError());

	global_best_final << <1, 1 >> > (dev_swarm, global_best_after_reduce, blocks_reduce);
	CSC(cudaGetLastError());

	CSC(cudaDeviceSynchronize());
	CSC(cudaGraphicsUnmapResources(1, &res, 0));

	CSC(cudaEventRecord(stop, 0));
	CSC(cudaEventSynchronize(stop));
	CSC(cudaEventElapsedTime(&time, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));
	printf("%.4f\n", time);

	glutPostRedisplay();
}

void display() {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

void MyKeyboardFunc(unsigned char Key, int x, int y)
{
	switch (Key)
	{
	case 27:
		CSC(cudaGraphicsUnregisterResource(res));
		glBindBuffer(1, vbo);
		glDeleteBuffers(1, &vbo);
		exit(0);
		break;
	case 'q':
		scale_x += 20;
		scale_y = scale_x * height / width;
		break;
	case 'e':
		if (scale_x > 30)
		{
			scale_x -= 20;
			scale_y = scale_x * height / width;
		}
		break;
	};
}

int main(int argc, char** argv)
{
	CSC(cudaMalloc(&dev_swarm, sizeof(particle) * (ceil(particle_cnt / (double)THREADS))  * THREADS));
	CSC(cudaMalloc(&dev_states, sizeof(curandState) * (ceil(particle_cnt / (double)THREADS)) * THREADS));
	CSC(cudaMalloc(&global_best_after_reduce, sizeof(double2) * ceil(particle_cnt / (double)THREADS)));
	CSC(cudaMalloc(&arr_max_after_reduce_dev, sizeof(double) * blocks_reduce));
	CSC(cudaMalloc(&arr_min_after_reduce_dev, sizeof(double) * blocks_reduce));

	swarm_start << <blocks1D, threads1D >> > (dev_swarm, particle_cnt, dev_states);
	CSC(cudaGetLastError());

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("YakimovichCP");

	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(MyKeyboardFunc);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);

	glewInit();

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);

	CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));

	glutMainLoop();

	return 0;
}