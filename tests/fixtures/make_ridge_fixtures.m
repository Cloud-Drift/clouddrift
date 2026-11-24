% MAKE_RIDGE_FIXTURES  Reference output from jLab for the CloudDrift ridge analysis port.
%
% Run this once, with jLab on the MATLAB path, from anywhere:
%
%     >> addpath(genpath('/path/to/jLab'))
%     >> run('/path/to/clouddrift/tests/fixtures/make_ridge_fixtures.m')
%
% It reads data/fixtures/ridge_fixture_inputs.mat, which is written by the Python side so
% that both languages analyse the identical numbers, and writes
% data/fixtures/ridge_fixture_outputs.mat, which the Python tests compare against.
%
% Three cases, chosen to isolate the layers of the port:
%
%   1. wavetrans   the wavelet transform itself. If this disagrees, everything downstream
%                  does, so it is checked first and separately.
%   2. ridgewalk   the ridge detection and chaining, on the transform computed here. This
%                  is the layer that has diverged in practice: the Python port uses a
%                  different chaining rule from jLab's.
%   3. eddyridges  the whole eddy pipeline on GDP drifter 44000, including the ellipse
%                  properties, which is what a census would report.
%
% Nothing is plotted and nothing else is written.

clear variables

if isempty(which('ridgewalk'))
    error('jLab is not on the MATLAB path: run addpath(genpath(''/path/to/jLab'')) first.')
end

here = fileparts(mfilename('fullpath'));
repository = fullfile(here, '..', '..');
input_file = fullfile(repository, 'data', 'fixtures', 'ridge_fixture_inputs.mat');
output_file = fullfile(repository, 'data', 'fixtures', 'ridge_fixture_outputs.mat');

if exist(input_file, 'file') ~= 2
    error('input file not found: %s', input_file)
end
load(input_file)
fprintf('read %s\n', input_file);
fprintf('  synthetic signal: %d samples, %d frequencies\n', length(x), length(fs));
fprintf('  drifter: %d points\n', length(lat));

jlab_version = 'unknown';
if exist('jlab_version', 'file') == 2
    jlab_version = evalc('jlab_version');
end
matlab_version = version();

%% 1. the wavelet transform, bandpass normalized, mirror boundary conditions
[wx, wy] = wavetrans(x, y, {1, ga, be, fs, 'bandpass'}, 'mirror');
fprintf('wavetrans: %d by %d transform\n', size(wx, 1), size(wx, 2));

%% 2. ridge analysis of the bivariate transform
% The components of a multivariate transform are passed as separate arguments, not stacked
% along a third dimension: ridgewalk collects the leading arguments that share the size of
% the first one and treats them as the components of one transform vector. The ridges are
% then joint ridges of the transform vector norm, as in Lilly and Olhede (2012).
[wxr, wyr, ir, jr, omega] = ridgewalk(dt, wx, wy, fs, P, M, RHO);
ridge_count = 1 + sum(isnan(ir));
fprintf('ridgewalk: %d ridge points in %d ridges\n', sum(~isnan(ir)), ridge_count);

%% 3. the eddy pipeline on the drifter
% Wrapped so that a failure here still leaves cases 1 and 2 saved.
eddy = struct();
eddy_error = '';
try
% One-sided ridges, the frequency band expressed as a ratio to the Coriolis frequency.
eddies = eddyridges(num, lat, lon, fmax, fmin, P, M, RHO);

% The output is a struct of cell arrays, one cell per ridge. Flatten it into ragged arrays
% plus a rowsize vector, which is how CloudDrift stores the same thing, so that the Python
% side does not have to unpack MATLAB cells.
fields = {'num', 'lat', 'lon', 'latres', 'lonres', 'kappa', 'xi', 'theta', 'phi', ...
          'omega', 'upsilon', 'chi', 'R', 'V', 'ro', 'ir', 'jr', 'kr'};
eddy = struct();
for k = 1:length(fields)
    name = fields{k};
    if isfield(eddies, name)
        column = eddies.(name);
        if iscell(column)
            eddy.(name) = cell2col(column);
        else
            eddy.(name) = column;
        end
    end
end
if isfield(eddies, 'len')
    eddy.len = eddies.len;
end
if isfield(eddies, 'num') && iscell(eddies.num)
    eddy.rowsize = cellfun(@length, eddies.num);
    fprintf('eddyridges: %d ridges, %d points\n', length(eddy.rowsize), sum(eddy.rowsize));
end
eddy.zhat = cell2col(eddies.zhat);
catch failure
    eddy_error = failure.message;
    fprintf(2, 'eddyridges failed: %s\n', eddy_error);
end

save(output_file, 'wx', 'wy', 'wxr', 'wyr', 'ir', 'jr', 'omega', 'eddy', 'eddy_error', ...
     'jlab_version', 'matlab_version', 'fs', 'ga', 'be', 'P', 'M', 'RHO', ...
     'fmax', 'fmin', 'dt', '-v7.3')
fprintf('wrote %s\n', output_file);
