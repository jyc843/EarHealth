function [filelist] = file_read(path_directory) 
 % path_directory='../human_data'; 
 % Pls note the format of files,change it as required
%   ls=append(ls,element);
filelist = [];
 original_files=dir([path_directory '/*.wav']); 
 for k=1:length(original_files)
    filename=[path_directory '/' original_files(k).name];
    filelist{k} = filename;
 end
end