load('testIOUallresult.mat');
temp = struct2cell(IOUallresult);
temp = temp(6,1,:);
temp = cell2mat(temp);
top5 = sum(temp);
top5rate = top5/num

temp1 = struct2cell(IOUallresult);
temp1 = temp1(7,1,:);
temp1 = cell2mat(temp1);
top1 = sum(temp1);
top1rate = top1/num