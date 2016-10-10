#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 computeDiffuseDirection(
glm::vec3& normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec3 computeReflectiveDirection(
const glm::vec3& normal, const glm::vec3& incident)
{
	const float cosI = glm::dot(normal, incident);
	return incident - 2.0f * cosI * normal;
}

__host__ __device__
glm::vec3 computeRefractiveDirection(
const glm::vec3& normal, const glm::vec3& incident,
const float ni, const float nt)
{
	// Uses Snell's Law:
	const float n = ni / nt;
	const float cosI = -glm::dot(normal, incident);
	float sinT2 = n * n * (1.0f - cosI * cosI);
	if (sinT2 > 1.0) { return computeReflectiveDirection(normal, incident); } //Hack
	const float cosT = sqrt(1.0 - sinT2);
	return n * incident + (n * cosI - cosT) * normal;
}

__host__ __device__
float computeReflectance(
const glm::vec3& normal, const glm::vec3& incident,
const float ni, const float nt)
{
	// From http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
	const float n = ni / nt;
	const float cosI = -glm::dot(normal, incident);
	float sinT2 = n * n * (1.0f - cosI * cosI);
	if (sinT2 > 1.0) { return 1.0f; } //Hack
	const float cosT = sqrt(1.0 - sinT2);
	const float rOrth = (ni * cosI - nt * cosT) / (ni * cosI + nt * cosT);
	const float rPar = (nt * cosI - ni * cosT) / (nt * cosI + ni * cosT);
	return (rOrth * rOrth + rPar * rPar) / 2.0f;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
		glm::vec3 eye,
        const Material &m,
        thrust::default_random_engine &rng) 
{
	Ray& outRay			= pathSegment.ray;
	glm::vec3& outColor	= pathSegment.color;

	const bool isInwardsRay = (glm::dot(normal, outRay.direction) < 0.0f);
	if (!m.hasReflective && !m.hasRefractive) // Diffuse
	{
		outRay.direction	= computeDiffuseDirection(normal, rng);
		pathSegment.color *= glm::abs(glm::dot(outRay.direction, normal)) * m.color;
	}
	else
	{
		float ni = 1.0f, nt = 1.0f;
		{
			if (isInwardsRay)
			{
				ni = 1.0f;
				nt = m.indexOfRefraction;
			}
			else
			{
				ni = m.indexOfRefraction;
				nt = 1.0f;
			}
		}

		float reflectance = computeReflectance( normal, outRay.direction, ni, nt );
		if (m.hasReflective)
		{
			outRay.direction = glm::reflect(pathSegment.ray.direction, normal);
			//pathSegment.color *= m.color;
			pathSegment.color *= glm::abs(glm::dot(outRay.direction, normal)) * m.color;
		}
		else if (m.hasRefractive)
		{
			outRay.direction = computeRefractiveDirection(normal, pathSegment.ray.direction, ni, nt);
			outColor *= m.color * (1.0f - reflectance);
		}
	}

	float pathDir = isInwardsRay ? -1 : 1;

	const float MY_EPSILON = 1e-4f;
	outRay.origin = intersect + pathDir * normal * MY_EPSILON;
}
