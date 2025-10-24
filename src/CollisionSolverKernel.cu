#include "ParticleKernels.cuh"

__device__ int gridCounts[GRID_WIDTH][GRID_HEIGHT][GRID_DEPTH];
__device__ int gridIndices[GRID_WIDTH][GRID_HEIGHT][GRID_DEPTH][MAX_PER_CELL];
__device__ float3 displacements[DISPLACEMENT_ARRAY_SIZE];

__device__ int3 getGridCell(float x, float y, float z) {
    int gx = min(max(int(x / GRID_SIZE_GPU), 0), GRID_WIDTH - 1);
    int gy = min(max(int(y / GRID_SIZE_GPU), 0), GRID_HEIGHT - 1);
    int gz = min(max(int(z / GRID_SIZE_GPU), 0), GRID_DEPTH - 1);
    return make_int3(gx, gy, gz);
}

__global__ void resetGridKernel3D() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < GRID_WIDTH && y < GRID_HEIGHT && z < GRID_DEPTH) {
        gridCounts[x][y][z] = 0;
        for (int k = 0; k < MAX_PER_CELL; k++)
            gridIndices[x][y][z][k] = -1;
    }
}

void resetGrid3D() {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((GRID_WIDTH + blockSize.x - 1) / blockSize.x,
        (GRID_HEIGHT + blockSize.y - 1) / blockSize.y,
        (GRID_DEPTH + blockSize.z - 1) / blockSize.z);
    resetGridKernel3D<<<gridSize, blockSize>>>();
}   
__global__ void buildGridKernel3D(VerletObjectCUDA* particles, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int3 cell = getGridCell(particles[i].current_position.x,
        particles[i].current_position.y,
        particles[i].current_position.z);

    int idx = atomicAdd(&gridCounts[cell.x][cell.y][cell.z], 1);
    if (idx < MAX_PER_CELL)
        gridIndices[cell.x][cell.y][cell.z][idx] = i;
}

__global__ void solveCollisionByGridKernel3D(VerletObjectCUDA* particles, int N,
    float response_coef, float attraction_coef, float repulsion_coef)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    displacements[i] = make_float3(0.0f, 0.0f, 0.0f);
    VerletObjectCUDA& obj = particles[i];
    int3 cell = getGridCell(obj.current_position.x, obj.current_position.y, obj.current_position.z);

    // Check neighbors in 3x3x3 cube
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int nx = cell.x + dx;
                int ny = cell.y + dy;
                int nz = cell.z + dz;

                if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT || nz < 0 || nz >= GRID_DEPTH)
                    continue;

                int count = gridCounts[nx][ny][nz];
                for (int k = 0; k < count; k++) {
                    int neighborIdx = gridIndices[nx][ny][nz][k];
                    if (neighborIdx == i || neighborIdx < 0) continue;

                    VerletObjectCUDA& neighbor = particles[neighborIdx];

                    float3 v = make_float3(obj.current_position.x - neighbor.current_position.x,
                        obj.current_position.y - neighbor.current_position.y,
                        obj.current_position.z - neighbor.current_position.z);

                    float dist2 = v.x * v.x + v.y * v.y + v.z * v.z;
                    float min_dist = obj.radius + neighbor.radius;
                    float r_min = 3.0f * obj.radius;

                    if (dist2 < r_min * r_min) {
                        float dist = sqrtf(dist2);
                        if (dist < 1e-6f) continue;

                        float3 n = make_float3(v.x / dist, v.y / dist, v.z / dist);
                        float delta = 0.5f * response_coef * (dist - min_dist);

                        float mass_ratio = obj.radius / min_dist;

                      

                        // collision response
                        if (dist2 < min_dist * min_dist) {
                            displacements[i].x -= n.x * delta * mass_ratio;
                            displacements[i].y -= n.y * delta * mass_ratio;
                            displacements[i].z -= n.z * delta * mass_ratio;
                        }
                    }
                }
            }
        }
    }
}

__global__ void applyDisplacementsKernel3D(VerletObjectCUDA* particles, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    particles[i].current_position.x += displacements[i].x;
    particles[i].current_position.y += displacements[i].y;
    particles[i].current_position.z += displacements[i].z;
}


__global__ void applyBoundaryCollisionKernel(VerletObjectCUDA* particles,
    uint64_t N,
    uint16_t boxWidth,
    uint16_t boxHeight,
    uint16_t boxDepth,
    float restitution)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    VerletObjectCUDA& p = particles[i];

    float3 pos = p.current_position;
    float3 vel = make_float3(pos.x - p.old_position.x,
        pos.y - p.old_position.y,
        pos.z - p.old_position.z);

    // X-axis (Left/Right)
    if (pos.x < p.radius) {
        pos.x = p.radius;
        vel.x = -vel.x * restitution;
    }
    else if (pos.x + p.radius > boxWidth) {
        pos.x = boxWidth - p.radius;
        vel.x = -vel.x * restitution;
    }

    // Y-axis (Bottom/Top)
    if (pos.y < p.radius) {
        pos.y = p.radius;
        vel.y = -vel.y * restitution;
    }
   

    // Z-axis (Front/Back)
    if (pos.z < p.radius) {
        pos.z = p.radius;
        vel.z = -vel.z * restitution;
    }
    else if (pos.z + p.radius > boxDepth) {
        pos.z = boxDepth - p.radius;
        vel.z = -vel.z * restitution;
    }

    // Update particle positions
    p.current_position = pos;
    p.old_position.x = pos.x - vel.x;
    p.old_position.y = pos.y - vel.y;
    p.old_position.z = pos.z - vel.z;
}
