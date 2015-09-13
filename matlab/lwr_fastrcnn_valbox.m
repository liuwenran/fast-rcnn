%% fast-rcnn config
[folder, name, ext] = fileparts(mfilename('fullpath'));

caffe_path = fullfile(folder, '..', 'caffe-fast-rcnn', 'matlab', 'caffe');
% caffe_path = 'external/caffe/matlab/caffe';

addpath(caffe_path);

use_gpu = true;
% You can try other models here:
def = fullfile(folder, '..', 'models', 'VGG16', 'test.prototxt');;
net = fullfile(folder, '..', 'data', 'fast_rcnn_models', ...
               'vgg16_fast_rcnn_iter_40000.caffemodel');
model = fast_rcnn_load_net(def, net, use_gpu);

%% boxfile and imagefile config
addpath(genpath('/data/VOC2007/VOCdevkit'));
addpath(genpath('/data/VOC2007/VOCdevkit/VOCcode'));

valanopath = '/data/VOC2007/VOCdevkit/VOC2007/Annotations';
flist = dir(valanopath);

num = 1;
imgNum = 4952;
bboxNum = 14976;
s=struct('fname',[],'bbox',[],'truelabel',[],'top5label',[],'top5scores',[],'correct5',false,'correct1',false ,'groundTruthRank',0);
IOUallresult = s;
IOUallresult(bboxNum) = s;
logFile=zeros(imgNum,5);
newlocation = [];

fprintf('Begin testing\n');
VOCinit;
% labelVector = initLabel();
% load 'allValid.mat' allValid;
load('/home/liuwr/rcnn/data/selective_search_data/voc_2007_test.mat');
load('/home/liuwr/rcnn/MaxIOUofDataSetBySS/VOC2007/test/voc2007maxIOU.mat');

originBoxes = boxes;
for i=1:length(images)
    name = cell2mat(images(i));
    filepath = [valanopath,'/',name,'.xml'];
    rec = VOCreadrecxml(filepath);
    
    imfile = ['/data/VOC2007/VOCdevkit/VOC2007/JPEGImages/' name '.jpg'];
    im = imread(imfile);
    
    channel = size(im,3);
    if channel == 1
        im = repmat(im, [1 1 3]);
    end
    
    numr5 = 0;
    numr1 = 0;
    numBoxes=length(rec.objects);
    
    for m=1:numBoxes  % test all objects
        
        IOUallresult(num).fname = rec.filename;
        %         IOUallresult(num).bbox = rec.objects(m).bbox;
        boxIndex = maxIOUmat(i).boxIndex(m);
        boxCandidate = cell2mat(originBoxes(i));
        boxThis = boxCandidate(boxIndex,:);
        boxCal = [boxThis(2),boxThis(1),boxThis(4),boxThis(3)];
        IOUallresult(num).bbox = boxCal;
        %         IOUallresult(num).truelabel = rec.objects(m).label;
        IOUallresult(num).truelabel = initLabel(rec.objects(m).class);
        
        %         boxes = rec.objects(m).bbox;
        boxes = single(boxCal);
        dets = lwr_fast_rcnn_im_detect(model, im, boxes);
        
        [v,idx]=sortrows(cell2mat(dets),-5);
        IOUallresult(num).top5label = idx(1:5);
        IOUallresult(num).top5scores = v(1:5,5);
%         IOUallresult(num).groundTruthRank = find(idx==rec.objects(m).label);
        IOUallresult(num).groundTruthRank = find(idx==initLabel(rec.objects(m).class));
        IOUallresult(num).correct5 = (IOUallresult(num).groundTruthRank <= 5);
        IOUallresult(num).correct1= (IOUallresult(num).groundTruthRank == 1);
        
        newlocation = [newlocation;v(1,1:4)];        
        
        if IOUallresult(num).correct5
            numr5 = numr5+1;
        end
        if IOUallresult(num).correct1
            numr1 = numr1+1;
        end
        
        logFile(num,:)=[i, m, numBoxes, numr1, numr5];
        num = num+1;
    end

    fprintf('Now tested %d/%d, top 5 correct rate: %d/%d, top 1 correct rate: %d/%d\n', i, imgNum, numr5, numBoxes, numr1, numBoxes);
    if rem(i, 1500)==0
        save('logFile_temp.mat','logFile');
        save('testIOUallresult_temp.mat','IOUallresult');
        save('newlocation_temp.mat','newlocation');
    end
end
save('logFile.mat','logFile');
save('testIOUallresult.mat','IOUallresult');
save('newlocation.mat','newlocation');



