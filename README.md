# Evolutionary optimisation algorithms: application to the Optimal Image Collage Composition problem

## Author
Kazimieras Jasaitis

## Abstract
In the realm of computational problem-solving, deterministic algorithms often fall short in addressing complex, computationally intensive tasks. This thesis delves into one such challenge: the creation of image collages, where multiple images must be arranged on a fixed-size canvas without overlaps or unoccupied space, a problem classified as NP-Hard. The focus is on the application of Particle Swarm Optimization (PSO) and Harmony Search Optimization (HSO), both evolutionary optimization algorithms, in solving the Image Collage Composition problem.

The PSO and HSO algorithms are examined in detail, particularly in their application to the problem through the development of a unit test generator and the execution of performance tests. The unit test generator segments an image into `n` pieces, upon which the PSO algorithm is employed to find optimal or near-optimal placements on a canvas. The performance of PSO is critically evaluated using various metrics, leading to valuable insights and recommendations for potential enhancements and future research directions.

## Running the Algorithms

To run the Particle Swarm Optimization (PSO) and Harmony Search Optimization (HSO) algorithms, you need to set up your environment with the required packages. 

### Prerequisites

- Python version required: **3.12.0**
- Additional required packages:
  - `numpy`
  - `pillow`

Dependencies are specified in the `testingEnv.yaml` file. Ensure that all dependencies are installed before attempting to run the algorithms.

### Installation

1. **Create and activate a Conda environment** (Optional):
   ```bash
   conda env create -f testingEnv.yaml
   conda activate your-env-name

2. **Install required packages**:
   If you are not using a Conda environment or need to install packages manually:
   ```bash
   pip install numpy pillow
   ```
### Running the Algorithms

You can run the algorithms from the command line:

- For Particle Swarm Optimization:
  ```bash
  py pso.py
  ```

- For Harmony Search Optimization:
  ```bash
  py hso.py
  ```

## Keywords
- Particle Swarm Optimization (PSO)
- Harmony Search Optimization (HSO)
- Evolutionary Methods
- Optimization
- 2D Bin Packing Problem
- Image Collage Composition