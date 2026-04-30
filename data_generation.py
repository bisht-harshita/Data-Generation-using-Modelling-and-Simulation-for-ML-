"""
Data Generation using SimPy
===========================
This script simulates a Call Center environment to generate a synthetic dataset for Machine Learning.

Scenario: Call Center
---------------------
- Customers arrive at random intervals (Poisson process).
- A limited number of agents handle calls.
- Service time is random (Exponential distribution).
- Customers wait in a queue if no agents are available.

Parameters (Inputs):
1. Number of Agents (2-10)
2. Average Arrival Interval (1-10 minutes)
3. Average Service Time (5-20 minutes)

Target (Output):
- Average Wait Time (minutes) per simulation run.
"""

import simpy
import random
import pandas as pd
import numpy as np

# Simulation Configuration
NUM_SIMULATIONS = 1000  # Total number of data points to generate
SIMULATION_TIME = 480   # 8 hours shift (in minutes)

dataset = []

def call_center(env, num_agents, service_time, arrival_interval, wait_times):
    """Process for the Call Center Simulation"""
    call_center_resource = simpy.Resource(env, capacity=num_agents)

    def customer(env, name, service_time, wait_times):
        arrival_time = env.now
        
        with call_center_resource.request() as request:
            yield request  # Wait for an agent
            
            wait_time = env.now - arrival_time
            wait_times.append(wait_time)
            
            # Service time (Randomly varied around the average)
            # Ensure service time is non-negative
            actual_service_time = max(1, random.expovariate(1.0 / service_time))
            yield env.timeout(actual_service_time)

    # Customer Generator
    i = 0
    while True:
        yield env.timeout(random.expovariate(1.0 / arrival_interval))
        i += 1
        env.process(customer(env, f'Customer {i}', service_time, wait_times))

def run_simulation(num_agents, arrival_interval, service_time):
    """Runs a single simulation instance and returns the average wait time"""
    env = simpy.Environment()
    wait_times = []
    
    env.process(call_center(env, num_agents, service_time, arrival_interval, wait_times))
    env.run(until=SIMULATION_TIME)
    
    if len(wait_times) > 0:
        return np.mean(wait_times)
    else:
        return 0.0

print(f"Starting Data Generation ({NUM_SIMULATIONS} simulations)...")

for i in range(NUM_SIMULATIONS):
    # Step 3: Randomly sample parameters (Lower and Upper bounds)
    # Agents: 2 to 20
    num_agents = random.randint(2, 20)
    
    # Arrival Interval: 1 to 10 minutes (High traffic to Low traffic)
    arrival_interval = random.uniform(1.0, 10.0)
    
    # Service Time: 5 to 30 minutes (Short calls to Long calls)
    service_time = random.uniform(5.0, 30.0)
    
    # Step 4: Run Simulation
    avg_wait = run_simulation(num_agents, arrival_interval, service_time)
    
    # Record Data
    dataset.append({
        'Num_Agents': num_agents,
        'Arrival_Interval': arrival_interval,
        'Service_Time': service_time,
        'Average_Wait_Time': avg_wait  # Target Variable
    })
    
    if (i+1) % 100 == 0:
        print(f"Generated {i+1}/{NUM_SIMULATIONS} samples...")

# Step 5: Save Dataset
df = pd.DataFrame(dataset)
df.to_csv('simulation_data.csv', index=False)
print("Data Generation Complete. Saved to 'simulation_data.csv'.")
print("\nSample Data:")
print(df.head())
