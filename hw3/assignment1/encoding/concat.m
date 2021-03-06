clear;
clc;
cd('..')
 
% load the mat files and concatenate them
bTest = false;
sift_dir = '.\results\sift\';
dir_list = dir(sift_dir);
dir_num = length(dir_list);
data_path = '.\results\siftLD\siftLD.mat';
class_path = '.\results\siftLD\classes.mat';
label_path = '.\results\siftLD\label.mat';
numLD_path = '.\results\siftLD\numLD.mat';
files_num = 0;
%% concatenate
sprintf('开始遍历local descriptor，共%d个文件夹',dir_num);
load(numLD_path,'numLD'); % the local descriptor number in a figure
lengthLD = 128; % the length of a local descriptor
lds = zeros(lengthLD,sum(numLD));
file_nums = zeros(dir_num-2,1); % use this arr to remember the class
k = 1;
cursor = 1;
for i= 3:dir_num % omit '.' and '..'
    sprintf('第%d组图片%s',i-2,dir_list(i).name);
    if dir_list(i).isdir
        dir_path = [sift_dir,dir_list(i).name,'\'];
        file_list = dir(dir_path);
        if bTest
            file_nums(i-2) = 10;
        else
            file_nums(i-2) = length(file_list)-2;
        end
        files_num = files_num + file_nums(i-2);
        for j = 1:file_nums(i-2)
            file_path = [dir_path,file_list(j+2).name];
            load(file_path,'d');
            lds(:,cursor:cursor+numLD(k)-1) = d;
            cursor = cursor+numLD(k);
            k = k+1;
        end
    end
end
%% build the label file
labels = zeros(files_num,1);
cursor = 1;
for i = 1:length(file_nums)
    labels(cursor:cursor+file_nums(i)-1) = i;
    cursor = cursor+file_nums(i);
end
%% save the results
save(data_path,'lds')
save(class_path,'file_nums')
save(label_path,'labels')