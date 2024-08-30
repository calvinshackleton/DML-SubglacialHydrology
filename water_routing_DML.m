%--------------------------------------------------------------------------
% Subglacial_Hydrology_Dome_Fuji.m   ***Matlab Script***
%   - Use topotools functions to simulate hydrology 
%--------------------------------------------------------------------------
%clear variables

function water_routing(folder)
    param= readcell([folder 'hydrology_params.txt'],'NumHeaderLines',1);
    param=cell2struct(param(:,2),param(:,1)); % struct for parameters

%--------------------------------------------------------------------------
% Run TopoTools hydrology analysis tools
    DEM= GRIDobj(param.file);  % create DEM object from .tif
    
    if param.elevate_minima_loops == 1
        DEMe= elevateminima(DEM, param.max_elev_min_area); % fills e.g. single pixel sinks 
    elseif param.elevate_minima_loops > 1
        DEMe= DEM;
        for i= 1:param.elevate_minima_loops
            DEMe= elevateminima(DEMe, param.max_elev_min_area); % fills sinks in loop
        end
    elseif param.elevate_minima_loops == 0  % No elevate minima tools run
        DEMe= DEM;
    end

    DEMf= fillsinks(DEMe,param.max_lake_level); % Fill sinks 
    lakes= DEMf-DEMe;   % locate filled sinks i.e. potential lakes
    lakes.Z(lakes.Z<param.min_lake_level)=NaN; % -9999 or NaN
    
    FD= FLOWobj(DEMf);  
    A= flowacc(FD);
    S= STREAMobj(FD,'minarea', param.upstream_area); % minarea= min. upstream area to initiate river
    C= curvature(DEMf);  % returns second derivative (curvature) of the DEM (alt find lakes)
    SO= streamorder(FD,flowacc(FD)>param.upstream_area);  % Strahler stream order ,'type',2
    D= drainagebasins(FD);
%--------------------------------------------------------------------------
% Save output to file
    GRIDobj2geotiff(DEM, [folder 'dem.tif'])
    GRIDobj2geotiff(DEMe, [folder 'deme_fill.tif'])
    GRIDobj2geotiff(DEMf, [folder 'demf_fill.tif'])
    GRIDobj2geotiff(lakes,[folder 'lakes.tif'])
    GRIDobj2geotiff(SO, [folder 'strahler_streams.tif'])
    GRIDobj2geotiff(A, [folder 'flow_accumulation.tif'])
%     GRIDobj2geotiff(elakes,[folder 'elakes.tif'])
%     GRIDobj2geotiff(clakes,[folder 'qcarved_lakes.tif'])
    GRIDobj2geotiff(D,[folder 'watersheds.tif'])
    GRIDobj2geotiff(C,[folder 'curvature.tif'])
    streams= STREAMobj2mapstruct(S);
    shapewrite(streams,[folder 'streams.shp'])


    % Other useful functions/methods
    % e.g. write shapefile for drainage basins/watersheds
    if run_misc_work==1
        outdir='S:\DronningMaudLand - Topographic uncertainty\_results_\hydro_03-24\watersheds_k_1\';
        basePath= 'S:\DronningMaudLand - Topographic uncertainty\_results_\DML_v3.4\';
        for k=0:49
            folderName = sprintf('%03d', k);
            % D= GRIDobj(fullfile(basePath,folderName,'k_1\watersheds.tif'));
            DEMf= GRIDobj(fullfile(basePath,folderName,'k_1\demf_fill.tif'));
            % SO= GRIDobj(fullfile(basePath,folderName,'k_1\strahler_streams.tif'));
            % FA= GRIDobj(fullfile(basePath,folderName,'k_1\flow_accumulation.tif'));
    
            FD= FLOWobj(DEMf); 
            S= STREAMobj(FD,'minarea', 2000);
            D = drainagebasins(FD,S); % Drainage basins for streams over threshold area
            % D = drainagebasins(FD,SO,3); % Drainage basins for ordered streams
            Dshp= GRIDobj2polygon(D);
    
            GRIDobj2geotiff(D,[outdir folderName '_watershed.tif'])
            shapewrite(Dshp,[outdir folderName '_basins.shp'])
    
            % figure; imagesc(D); colorbar;
        end
    end
end