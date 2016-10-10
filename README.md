# University of Pennsylvania, CIS 565: GPU Programming and Architecture.
Project 3 CUDA: Path Tracer
====================

## User resources
- **Name:** David Grosman.
- **Tested on:** Microsoft Windows 7 Professional, i7-5600U @ 2.6GHz, 256GB, GeForce 840M (Personal laptop).

## Project description
This Project's purpose was to gain some experience with writing Graphics code that would benefit as much as possible from CUDA as possible. In fact, a path tracer is a very useful type of application to run on Cuda since each ray processed by the application can be done in a separate thread. Furthermore, computations such as intersection-testing and pixel coloring are very intensive and are thus most usefully done on GPU. Furthermore, there is no memory bandwidth from CPU to GPU to pass on the pixel buffer as is usually the case in a CPU implementation.
In this project, I have implemented several key features such as:

1. Diffuse shading.
2. Caching Initial rays.
3. Stream compaction on terminated paths during each iteration.
4. Sort Rays by material type they are intersecting with.
5. Specular reflection and refraction.
6. Depth of Field.
7. Jittered, Halton and Sobel Pseudo-Random generators to improve the cosine weighted hemisphere generated when computing the diffuse component.



###Shading, 
I first implemented my efficient scan version following the class slides closely but I was unsatisfied with its performance -- it was around 8 times slower than my cpu approach.
![](img/RefrAndRefl.JPG)



---
### Performance Analysis
Note that the following statistics have been captured by calling the given functions 1000 times (with the default parameters given when starting the project) and averaging the results.
Please note that I used CUDA events for timing GPU code and I did not include any initial/final memory operations (cudaMalloc, cudaMemcpy) in your performance measurements, for comparability.
![](images/ScanRuntimePerformanceGivenArraySize.JPG)

It is interesting to notice that the CPU version is the fastest. It is most probably due to the fact that the algorithm on CPU is O(n) and accessing contiguous memory on CPU is very fast compared to GPU. The performance time on GPU decreases much faster given a bigger array size. It confirms that memory access is the GPU's performance main bottleneck.
It is nice to see that the efficient-scan performs better than the Naive implementation, even though it doesn't outperform Thrust's version which might include more efficient tricks.

![](images/CompactRuntimePerformanceGivenArraySize.JPG)

