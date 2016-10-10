#pragma once

class MyTimer;

class Timer
{
public:
	Timer();
	~Timer();

public:
	static void initializeTimer();
	static void shutdownTimer();

public:
	static void resetTimer(bool useGPU = true);
	static void playTimer();
	static void pauseTimer();
	static void printTimer(const char* timerHeader, float timerFactor);

private:
	static MyTimer* m_myTimer;
};

/**
* Handy-dandy hash function that provides seeds for random number generation.
*/
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>

struct PRNGenerator
{
	enum eType
	{
		None,
		Thrust,
		Stratified,
		Halton,
		Sodel,
	};

	static void initializeSystem(eType prngType, int numVals);
	static void shutdownSystem(void);

	static eType getType()		{ return m_type; };
	static size_t getNumVals()	{ return m_numVals; };
	static float* getVals()		{ return m_vals; };

private:

	static eType m_type;
	static size_t m_numVals;
	static float* m_vals;
};

struct PRNG
{
	__host__ __device__ PRNG(PRNGenerator::eType prngType, int numVals, float* vals);
	__host__ __device__ ~PRNG();

	__host__ __device__ void setSeed(int iter, int index, int depth);
	__host__ __device__	float getNextVal01();

	PRNGenerator::eType m_type;
	size_t m_numVals;
	float* m_vals;

	size_t m_nextRandomIdx;
	thrust::default_random_engine m_rng;
	thrust::uniform_real_distribution<float> m_u01;
};
