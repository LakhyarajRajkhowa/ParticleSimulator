#include "CollisionSolverKernel.cuh"


__device__ int gridCounts[GRID_WIDTH][GRID_HEIGHT];
__device__ int gridIndices[GRID_WIDTH][GRID_HEIGHT][MAX_PER_CELL];
__device__ float2 displacements[DISPLACEMENT_ARRAY_SIZE];

__device__ int2 getGridCell(float x, float y) {
    int gx = min(max(int(x / GRID_SIZE_GPU), 0), GRID_WIDTH - 1);
    int gy = min(max(int(y / GRID_SIZE_GPU), 0), GRID_HEIGHT - 1);
    return make_int2(gx, gy);
}

__global__ void resetGridKernel() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < GRID_WIDTH && y < GRID_HEIGHT) {
        gridCounts[x][y] = 0;
        for (int k = 0; k < MAX_PER_CELL; k++)
            gridIndices[x][y][k] = -1;
    }
}
void resetGridGPU() {
    dim3 threads(16, 16);
    dim3 blocks((GRID_WIDTH + threads.x - 1) / threads.x,
        (GRID_HEIGHT + threads.y - 1) / threads.y);
    resetGridKernel << <blocks, threads >> > ();
}
__global__ void buildGridKernel(VerletObjectCUDA* particles, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int2 cell = getGridCell(particles[i].current_position.x,
        particles[i].current_position.y);

    int idx = atomicAdd(&gridCounts[cell.x][cell.y], 1);
    if (idx < MAX_PER_CELL)
        gridIndices[cell.x][cell.y][idx] = i;
}

void buildGridGPU(VerletObjectCUDA* d_particles, int N) {
    int threads = 64;
    int blocks = (N + threads - 1) / threads;
    buildGridKernel << <blocks, threads >> > (d_particles, N);
}

__global__ void solveCollisionByGridKernel(VerletObjectCUDA* particles, int N, float response_coef, float attraction_coef, float repulsion_coef) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    displacements[i] = make_float2(0.0f, 0.0f);

    VerletObjectCUDA& obj = particles[i];
    int2 cell = getGridCell(obj.current_position.x, obj.current_position.y);

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int nx = cell.x + dx;
            int ny = cell.y + dy;
            if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT) continue;

            int count = gridCounts[nx][ny];
            for (int k = 0; k < count; k++) {
                int neighborIdx = gridIndices[nx][ny][k];
                if (neighborIdx == i || neighborIdx < 0) continue;

                VerletObjectCUDA& neighbor = particles[neighborIdx];

                float2 v = make_float2(obj.current_position.x - neighbor.current_position.x,
                    obj.current_position.y - neighbor.current_position.y);
                float dist2 = v.x * v.x + v.y * v.y;
                float min_dist = obj.radius + neighbor.radius;

                float r_min = 3 * obj.radius;

                if (dist2 < r_min * r_min) {
                    float dist = sqrtf(dist2);
                    if (dist < 1e-6f) continue; // avoid division by zero

                    float2 n = make_float2(v.x / dist, v.y / dist);
                    float delta = 0.5f * response_coef * (dist - min_dist);

					float attraction = attraction_coef * (dist - r_min) / dist;
					float repulsion = repulsion_coef * (dist - r_min) / dist;

                    float mass_ratio_obj = obj.radius / min_dist;
                    float mass_ratio_neighbor = neighbor.radius / min_dist;
                   
                    // molecular attraction
                    displacements[i].x += n.x * attraction * mass_ratio_obj;
                    displacements[i].y += n.y * attraction * mass_ratio_obj;

					// molecular repulsion
                    displacements[i].x -= n.x * repulsion * mass_ratio_obj;
                    displacements[i].y -= n.y * repulsion * mass_ratio_obj;

                    float max_vel = 1.0f;
					// collision response
                    if (dist2 < min_dist * min_dist) {
                        displacements[i].x -= (n.x * delta * mass_ratio_obj < max_vel)? (n.x * delta * mass_ratio_obj): max_vel;
                        displacements[i].y -= (n.y * delta * mass_ratio_obj < max_vel) ? (n.y * delta * mass_ratio_obj) : max_vel;

                    }
      
                }
            }
        }
    }

    
}

__global__ void applyDisplacementsKernel(VerletObjectCUDA* particles, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    particles[i].current_position.x += displacements[i].x;
    particles[i].current_position.y += displacements[i].y;
}

__global__ void applyBoundaryCollisionKernel(VerletObjectCUDA* particles, int N, int screenWidth, int screenHeight, float restitution)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    VerletObjectCUDA& p = particles[i];

    float3 pos = p.current_position;
    float2 vel = make_float2(p.current_position.x - p.old_position.x,
        p.current_position.y - p.old_position.y);

    // Left/right walls
    if (pos.x < p.radius) {
        pos.x = p.radius;
        vel.x = -vel.x * restitution;
    }
    else if (pos.x + p.radius > screenWidth) {
        pos.x = screenWidth - p.radius;
        vel.x = -vel.x * restitution;
    }

    // Top/bottom walls
    if (pos.y < p.radius) {
        pos.y = p.radius;
        vel.y = -vel.y * restitution;
    }
    else if (pos.y + p.radius > screenHeight) {
        pos.y = screenHeight - p.radius;
        vel.y = -vel.y * restitution;
    }

    p.current_position = pos;
    p.old_position.x = pos.x - vel.x;
    p.old_position.y = pos.y - vel.y;
}