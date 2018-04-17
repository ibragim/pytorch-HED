clear; clc;

path_to_pdollar = '../edges';
path_to_input = '/home/mameng/deeplearning/pytorch-HED/eval/test_HED';
path_to_output = '/home/mameng/deeplearning/pytorch-HED/eval/test_HED_nms';

addpath(genpath(path_to_pdollar));
mkdir(path_to_output);

iids = dir(fullfile(path_to_input, '*.png'));
for i = 1:length(iids)
    edge = imread(fullfile(path_to_input, iids(i).name));
    edge = 1-single(edge)/255;

    [Ox, Oy] = gradient2(convTri(edge, 4));
    [Oxx, ~] = gradient2(Ox);
    [Oxy, Oyy] = gradient2(Oy);
    O = mod(atan(Oyy .* sign(-Oxy) ./ (Oxx + 1e-5)), pi);
    % 2 for BSDS500 and Multi-cue datasets, 4 for NYUD dataset
    edge = edgesNms(edge, O, 2, 5, 1.01, 8);
    
    imwrite(edge, fullfile(path_to_output, [iids(i).name(1:end-4) '.png']));
    
end
