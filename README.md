#-----------------------------------------------------------------------------
# DML-SubglacialHydrology
#-----------------------------------------------------------------------------
This repository contains python code that simulates an ensemble of subglacial bed elevation grids and associated meltwater drainage in Dronning Maud Land, Antarctica. 

The analysis is used in the following publication:
"Evidence of active subglacial lakes under a slowly moving coastal region of the Antarctic Ice Sheet"
submitted to The Cryosphere by Arthur, et al.

Contact Calvin Shackleton (calvin.shackleton@npolar.no) for questions related to the stochastic simulated bed elevations and water routing analysis.
For other questions related to the publication, please contact Jennifer Arthur (jennifer.arthur@npolar.no)

#-----------------------------------------------------------------------------
The workflow for this analysis follows this general structure:
#-----------------------------------------------------------------------------
1. Use exp_variogram_DML.py to pre-process XYZ data and interactively calculate then model experimental variograms
2. Run SGSim_HPC_single.py to simulate bed elevation grids based on measurements and modelled variograms
	*Note: This script is formatted to run as many single processes in a HPC environment to generate the ensemble
3. Run subglacial_drainage_DML.py to calculate hydraulic potential and run water_routing_DML.m script in Matlab
4. Run ensemble_analysis_DML.py to output ensemble statistics and stream probability maps

#-----------------------------------------------------------------------------
