/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	// set number of particle
	num_particles = 100;
	weights.resize(num_particles);
	particles.resize(num_particles);

	random_device rd;
	default_random_engine gen(rd());

	// Create normal (Gaussian) distribution
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(x, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Initialize particles -
	for(int i = 0; i < num_particles; ++i){

		particles[i].id = i;
		particles[i].x  = dist_x(gen);
		particles[i].y  = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1.0;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Engine for later generation of particle
	default_random_engine gen;

	// Make distributions for adding noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // Different equations based on if yaw rate is zero or not

  for(int i = 0; i < num_particles; ++i){

	  if(abs(yaw_rate)!= 0)
	  {
		  // add measurement to particle 
		  particles[i].x += (velocity/yaw_rate)* (sin(particles[i].theta + (yaw_rate*delta_t)) -sin(particles[i].theta));
		  particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
          particles[i].theta += yaw_rate * delta_t;
	  } 
	  else 
	  {

		  // add measurement to particles
		  particles[i].x +=velocity*delta_t * cos(particles[i].theta);
		  particles[i].y +=velocity*delta_t * sin(particles[i].theta);		  	  
	  }

	// Add noise to the particles
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for( unsigned int i = 0; i< observations.size(); i++){
		// grab current observation
		LandmarkObs o = observations[i];

		// init minimum distance to maxinum possible
		double min_dist = numeric_limits<double>::max();
		// init id 
		int map_id = -1;

		for( unsigned int j = 0; j < predicted.size(); j++){
			// grap current prediction
			LandmarkObs p = predicted[i];
			
			double cur_dist = dist(o.x, o.y, p.x, p.y);

			if(cur_dist < min_dist){
				min_dist= cur_dist;
				map_id= p.id;
			}
		}
		// set observation's id to the nearest predicted landmark's id
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// for each particle..
	for(auto& p:particles){
		p.weight = 1.0;

		// step 1: collect valid landmarks
		vector<LandmarkObs> prediction;
		for(const auto& lm: map_landmarks.landmark_list){
			double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
		if(distance < sensor_range){
			prediction.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
		}
		}

		//step2: covert observation coordinates from vehicle to map
		vector<LandmarkObs> observations_map;
		double cos_theta = cos(p.theta);
		double sin_theta = sin(p.theta);

		for(const auto& obs: observations){
			LandmarkObs tmp;
			tmp.x = obs.x * cos_theta -obs.y * sin_theta + p.x;
			tmp.y = obs.x * sin_theta +obs.y*cos_theta +p.y;
			observations_map.push_back(tmp);
		}

		// Step 3: find landmark index for each observation
		dataAssociation(prediction, observations_map);

		// step 4: compute the particle's weight

		for(const auto& obs_m: observations_map){
			Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);
			double x_term = pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
      		double y_term = pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      		double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      		p.weight *=  w;
		}

		weights.push_back(p.weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	//generate distribution according to weights
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<>dist(weights.begin(), weights.end());

	// create resample particles
	vector<Particle> resample_particles;
	resample_particles.resize(num_particles);

	// resample the particles according to weights
	for(int i = 0; i < num_particles; i++){
		int idx = dist(gen);
		resample_particles[i] = particles[idx];
	}
	// assign the resampel particle to the previous particle
	particles = resample_particles;
	//clear the weight vector for the next round
	weights.clear();
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
