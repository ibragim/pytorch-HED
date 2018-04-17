model_name_lst = {'RESULTS_PATH'}

for i = 1:size(model_name_lst,2)
    tic;
    resDir ='/home/mameng/deeplearning/pytorch-HED/eval/test_HED_nms'; %fullfile('./NMS_RESULTS_FOLDER/',model_name_lst{i});
    fprintf('%s\n',resDir);
    gtDir = '/home/mameng/deeplearning/pytorch-HED/eval/test_gt';
    edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075);

    figure; edgesEvalPlot(resDir,'HED');
    toc
end