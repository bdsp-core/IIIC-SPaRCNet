function run_all()
% RUN_ALL  Regenerate every main-text & supplementary figure/table from the
% committed, de-identified proximal data in ./Data/. Run from code_for_figures/.
%
%   >> cd code_for_figures
%   >> run_all
%
% Each figure script loads its inputs from ./Data/<Figure*>/ and writes a PNG here.
% See ../REPRODUCE.md for the figure -> script -> input -> output map.

scripts = { ...
    'Table1_Splits', ...
    'Figure1_ROC', ...
    'Figure2_PR', ...
    'Figure3_UMAPs', ...
    'Figure4_samples_SZ_LPD', ...
    'Figure5_samples_GPD_LRDA', ...
    'Figure6_samples_GRDA_Other', ...
    'FigureS2_Flowchart', ...
    'FigureS3_SPspread', ...
    'FigureS5_UMAPspread', ...
    'FigureS8_IRR' };

nok = 0;
for i = 1:numel(scripts)
    fprintf('=== %s ===\n', scripts{i});
    nok = nok + run_one(scripts{i});   % isolated workspace so a script's `clear` is harmless
end
fprintf('\nrun_all: %d/%d scripts completed. PNGs written to code_for_figures/.\n', ...
        nok, numel(scripts));
end

function ok = run_one(name)
ok = 0;
try
    run([name '.m']);
    ok = 1;
catch e
    fprintf(2, 'FAILED %s: %s\n', name, e.message);
end
close all force;
end
