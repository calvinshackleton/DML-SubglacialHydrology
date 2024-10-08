# DML-SubglacialHydrology

This repository contains functions and workflows that simulates an ensemble of subglacial bed elevation grids using sequential Gaussian simulation, then predicts the routing of subglacial water for each simulated elevation grid in the ensemble. The analysis is set up to run in Dronning Maud Land, Antarctica.
The simulation step is formatted to run as many single tasks in a HPC environment, configured in a job submit script. 
Some of the geostatistical simulation tools are modified from or based on the [GStatSim](https://github.com/GatorGlaciology/GStatSim) python package, and subglacial meltwater drainage is predicted using flow routing tools from the [TopoToolbox](https://se.mathworks.com/matlabcentral/fileexchange/50124-topotoolbox) in Matlab.


Results are discussed in the following publication:<br>
"Evidence of active subglacial lakes under a slowly moving coastal region of the Antarctic Ice Sheet" <br>
*Submitted to* [The Cryosphere by Arthur, et al. 2024](https://egusphere.copernicus.org/preprints/2024/egusphere-2024-1704/#discussion)

Results are [available here](https://doi.org/10.21334/npolar.2024.b438191c), archived at the Norwegian Polar Data Center under: <br>
"*Ensemble analysis of potential subglacial meltwater streams in coastal Dronning Maud Land, Antarctica*",<br>
Shackleton, C., Matsuoka, K., Arthur, J., Moholdt, G., van Oostveen, J. (2024). *Norwegian Polar Institute*.<br>
https://doi.org/10.21334/npolar.2024.b438191c

## Usage

The analysis can be repeated by following this workflow:

1. Use [exp_variogram_DML.py](/exp_variogram_DML.py) to pre-process XYZ data and interactively calculate then model experimental variograms
2. Run [SGSim_HPC_single.py](/SGSim_HPC_single.py) to simulate bed elevation grids based on measurements and modelled variograms
	*Note: This script is formatted to run as many single processes in a HPC environment to generate the ensemble
3. Run [subglacial_drainage_DML.py](/subglacial_drainage_DML.py) to calculate hydraulic potential and run water_routing_DML.m script in Matlab
4. Run [ensemble_analysis_DML.py](/subglacial_drainage_DML.py) to output ensemble statistics and stream probability maps

## Contact
Contact Calvin Shackleton (calvin.shackleton@npolar.no) for questions relating to the simulated bed elevations and water routing analysis.
For other questions related to the publication, please contact Jennifer Arthur (jennifer.arthur@npolar.no)
