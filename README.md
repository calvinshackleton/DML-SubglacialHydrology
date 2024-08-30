# DML-SubglacialHydrology

This repository contains python code that simulates an ensemble of subglacial bed elevation grids using sequential Gaussian simulation. 
Some of the geostatistical simulation tools are modified from/based on the [GStatSim](https://github.com/GatorGlaciology/GStatSim) python package. 
Subglacial meltwater drainage associated with each simulated grids is predicted using flow routing tools from the [TopoToolbox](https://se.mathworks.com/matlabcentral/fileexchange/50124-topotoolbox) in Matlab.
The analysis is run in the Dronning Maud Land region, Antarctica, but can be adapted to run elsewhere.

The results are used in the following publication:
>"Evidence of active subglacial lakes under a slowly moving coastal region of the Antarctic Ice Sheet"

*Submitted to* [The Cryosphere by Arthur, et al. 2024](https://egusphere.copernicus.org/preprints/2024/egusphere-2024-1704/#discussion)



## Usage

The workflow for this analysis follows this general structure:

1. Use [exp_variogram_DML.py](/exp_variogram_DML.py) to pre-process XYZ data and interactively calculate then model experimental variograms
2. Run [SGSim_HPC_single.py](/SGSim_HPC_single.py) to simulate bed elevation grids based on measurements and modelled variograms
	*Note: This script is formatted to run as many single processes in a HPC environment to generate the ensemble
3. Run [subglacial_drainage_DML.py](/subglacial_drainage_DML.py) to calculate hydraulic potential and run water_routing_DML.m script in Matlab
4. Run [ensemble_analysis_DML.py](/subglacial_drainage_DML.py) to output ensemble statistics and stream probability maps

## Contact
Feel free to contact Calvin Shackleton (calvin.shackleton@npolar.no) for questions relating to the simulated bed elevations and water routing analysis.
For other questions related to the publication, please contact Jennifer Arthur (jennifer.arthur@npolar.no)
