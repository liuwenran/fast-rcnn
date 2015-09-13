function [ output_args ] = initLabel(class)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
 temp = {...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};
flag = 0;
output_args = 0;
for i=1:length(temp)
    if(strcmp(class,temp{i}))
        output_args = i;
        flag = 1;
        break;
    end
end
if(flag == 0)
    fprintf('wanla');
    output_args = 1;
end

end

