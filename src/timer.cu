
#include <cstdio>
#include <chrono>

#include "timer.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
static void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

MyTimer* Timer::m_myTimer = NULL;

class MyTimer
{
public:
	using Clock = std::chrono::high_resolution_clock;
	using TimePoint = std::chrono::time_point<Clock>;

public:
	MyTimer()
	{
		m_refCount = 0;
		m_useGPU = true;
		m_elapsedTimeInms = 0.0f;

		cudaEventCreate(&m_start);
		cudaEventCreate(&m_stop);

		m_startTime = Clock::now();
		m_stopTime = Clock::now();
	}

	~MyTimer()
	{
		cudaEventDestroy(m_start);
		cudaEventDestroy(m_stop);
	}

public:

	void resetTimer(bool useGPU = true) 
	{
		m_useGPU = useGPU;
		m_elapsedTimeInms = 0.0f;
	}

	void playTimer()
	{
		if (m_refCount++ == 0)
		{
			if (m_useGPU)
			{
				cudaEventRecord(m_start);
			}
			else
			{
				m_startTime = Clock::now();
			}
		}
	}

	bool pauseTimer()
	{
		bool bPaused = false;
		if (--m_refCount == 0)
		{
			float newElapsedTime = 0.0f;
			if (m_useGPU)
			{
				cudaEventRecord(m_stop);
				cudaEventSynchronize(m_stop);
				cudaEventElapsedTime(&newElapsedTime, m_start, m_stop);
			}
			else
			{
				m_stopTime = Clock::now();
				newElapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(m_stopTime - m_startTime).count();
			}
			m_elapsedTimeInms += newElapsedTime;
			bPaused = true;
		}
		return bPaused;
	}

	float printTimer(const char* timerHeader, float timerFactor)
	{
		float elapsedTime = timerFactor * m_elapsedTimeInms;
		printf("%s - Elapsed Time:%f ms.\n", timerHeader, elapsedTime);
		return elapsedTime;
	}

private:
	size_t m_refCount;
	bool m_useGPU;
	float m_elapsedTimeInms;
private:
	cudaEvent_t m_start;
	cudaEvent_t m_stop;

private:
	TimePoint m_startTime;
	TimePoint m_stopTime;
};

Timer::Timer()
{
}

Timer::~Timer()
{
}

void Timer::initializeTimer()
{
	if (m_myTimer == NULL)
		m_myTimer = new MyTimer;
}

void Timer::shutdownTimer()
{
	if (m_myTimer != NULL)
		delete m_myTimer;
}

void Timer::resetTimer(bool useGPU)
{
	m_myTimer->resetTimer(useGPU);
}

void Timer::playTimer()
{
	m_myTimer->playTimer();
}

void Timer::pauseTimer()
{
	m_myTimer->pauseTimer();
}

void Timer::printTimer(const char* timerHeader, float timerFactor)
{
	m_myTimer->printTimer(timerHeader, timerFactor);
}


//-------------------------------
//-------------RNG TESTS---------
//-------------------------------

void xOrBinValues(const char* x, const char* y, char* z)
{
	int bitXIdx = 0; int bitYIdx = 0; int bitZIdx = 0;
	while (x[bitXIdx] != NULL || y[bitYIdx] != NULL)
	{
		int xBit = 0;
		if (x[bitXIdx] != NULL)
		{
			xBit = (x[bitXIdx] == '1') ? 1 : 0;
			bitXIdx++;
		}

		int yBit = 0;
		if (y[bitYIdx] != NULL)
		{
			yBit = (y[bitYIdx] == '1') ? 1 : 0;
			bitYIdx++;
		}
		z[bitZIdx++] = ((xBit ^ yBit) == 0) ? '0' : '1';
	}
	z[bitZIdx] = NULL;
}

int indexOfRightMostZeroBit(int bitMask)
{
	int bitIdx = 1;
	while (bitMask)
	{
		if ((bitMask & 0x1) == 0) break;
		bitMask >>= 1;
		++bitIdx;
	}
	return bitIdx;
}

int computeViInBinary(int mi, int i, char* outVi)
{
	outVi[i] = NULL;
	for (int iter = i - 1; iter >= 0; --iter, mi >>= 1)
	{
		char nextBin = (mi & 0x1 == 1) ? '1' : '0';
		outVi[iter] = nextBin;
	}
	return i;
}

float convertBinary01ToFloat(const char* bin)
{
	float val = 0;
	int idx = 0;
	while (bin[idx] != NULL)
	{
		int bit = (bin[idx] == '0') ? 0 : 1;
		val += bit * powf(2, -(idx + 1));
		idx++;
	}
	return val;
}

void preComputeMiTable(const int numMi, int* miBuffer)
{
	// From ALGORITHM 659 Implementing Sobol’s Quasirandom Sequence Generator.
	// Each mi is odd and mi < (1<<i) with 1 <= i <= d.
	const int polyD = 3;

	static const int primPoly[polyD + 1] = { 1, 0, 1, 1 };
	static int miCoeffs[polyD + 1];
	static int    miIdx[polyD + 1];
	{
		miCoeffs[polyD] = 1;
		miIdx[polyD] = -polyD;
		for (int i = 1; i < 4; ++i)
		{
			miCoeffs[i - 1] = (1 << i) * primPoly[i];
			miIdx[i - 1] = -i;
		}
	}

	miBuffer[0] = 1; miBuffer[1] = 3; miBuffer[2] = 7;

	for (int iter = 3; iter < numMi; ++iter)
	{
		int newMi = 0;
		for (int i = 0; i <= polyD; ++i)
		{
			int miNext = miCoeffs[i] * miBuffer[iter + miIdx[i]];
			newMi ^= miNext;
		}
		miBuffer[iter] = newMi;
	}
}

float generateNextSodelRNG()
{
	static size_t iter = 0;
	static int miT[8192];
	static char xPrev[512];

	if (iter == 0)
	{
		preComputeMiTable(8192, miT);
		computeViInBinary(0, 1, xPrev);
		++iter;
		return 0.0f;
	}

	char x[256];
	char v[256];

	int c = indexOfRightMostZeroBit(iter - 1);
	computeViInBinary(miT[c - 1], c, v);
	xOrBinValues(xPrev, v, x);
	std::memcpy(xPrev, x, 256 * sizeof(char));

	iter++;
	float retVal = convertBinary01ToFloat(x);
	return retVal;
}


////
int numInBase(int num10, int base, char* out)
{
	int idx = 0;
	if (num10 == 0) { out[0] = '0'; return 1; }
	while (num10)
	{
		out[idx++] = num10 % base + '0';
		num10 = num10 / base;
	}
	out[idx] = 0;
	return idx;
}

float generateNextHaltonRNG()
{
	static const int primeBase = 7;
	static int iter = 0;

	char numStr[512];
	int numDigits = numInBase(iter, primeBase, numStr);

	float retVal = 0.0f;
	for (int digit = 0; digit < numDigits; ++digit)
	{
		int num = numStr[digit] - '0';
		if (num != 0)
			retVal += num * powf(primeBase, -(digit + 1));
	}

	++iter;
	return retVal;
}

float generateNextStratifiedRNG()
{
	static const int GridSize = 64;
	static size_t iter = 0;
	static bool isInit = false;

	static thrust::default_random_engine rng;
	if (!isInit)
	{
		const int h = utilhash((1 << 31) | (GridSize << 22) | iter) ^ utilhash(iter);
		rng = thrust::default_random_engine(h);
		isInit = true;
	}
	thrust::uniform_real_distribution<float> u01(0, 1);

	const int gridIdx = iter % GridSize;
	const float i01Val = u01(rng);

	++iter;
	return gridIdx / (1.0f * (GridSize - 1)) + i01Val;
}
/////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

//PRNGenerator::eType PRNGenerator::m_type = PRNGenerator::None;
//static size_t m_numVals = 0;
//static float* m_vals = NULL;

PRNGenerator::eType PRNGenerator::m_type = PRNGenerator::None;
size_t PRNGenerator::m_numVals = 0;
float* PRNGenerator::m_vals = NULL;

void PRNGenerator::initializeSystem(eType prngType, int numVals)
{
	if (prngType == PRNGenerator::None || m_vals != NULL)
	{
		fprintf(stderr, "ERROR");
		return;
	}
	checkCUDAError("generate camera ray");
	m_type = prngType;
	m_numVals = numVals;

	cudaMalloc(&m_vals, numVals * sizeof(float));
	float* vals = new float[numVals]; memset(vals, 0, numVals* sizeof(float));

	checkCUDAError("generate camera ray");
	switch (m_type)
	{
	case PRNGenerator::Thrust:
		break;
	case PRNGenerator::Stratified:
		for (int i = 0; i < numVals; ++i)
		{
			vals[i] = generateNextStratifiedRNG();
		}
		break;
	case PRNGenerator::Halton:
		for (int i = 0; i < numVals; ++i)
		{
			vals[i] = generateNextHaltonRNG();
		}
		break;
	case PRNGenerator::Sodel:
		for (int i = 0; i < numVals; ++i)
		{
			checkCUDAError("generate camera ray");
			vals[i] = generateNextSodelRNG();
		}
		break;
	default:
		break;
	}

	cudaMemcpy(m_vals, vals, numVals * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	delete [] vals;
	checkCUDAError("generate camera ray");
}

void PRNGenerator::shutdownSystem()
{
	cudaFree(m_vals);
}

PRNG::PRNG(PRNGenerator::eType prngType, int numVals, float* vals)
: m_type(prngType),
  m_numVals(numVals), m_vals(vals)
{

}

__host__ __device__ void PRNG::setSeed(int iter, int index, int depth)
{
	const int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	switch (m_type)
	{
	case PRNGenerator::Thrust:
		m_rng = thrust::default_random_engine(h);
		break;
	case PRNGenerator::Stratified:
	case PRNGenerator::Halton:
	case PRNGenerator::Sodel:
	{
		const size_t absH = abs(h);
		m_nextRandomIdx = absH % m_numVals;
		break;
	}
	default:
		break;
	}
}

__host__ __device__ PRNG::~PRNG()
{
}

__host__ __device__ float PRNG::getNextVal01()
{
	float nextVal = 0.0f;
	switch (m_type)
	{
	case PRNGenerator::Thrust:
		nextVal = m_u01(m_rng);
		break;
	case PRNGenerator::Stratified:
	case PRNGenerator::Halton:
	case PRNGenerator::Sodel:
	{
		nextVal = m_vals[m_nextRandomIdx++];
		if (m_nextRandomIdx >= m_numVals)
		{
			m_nextRandomIdx = 0;
		}
	}
		break;
	default:
		break;
	}
	return nextVal;
}
