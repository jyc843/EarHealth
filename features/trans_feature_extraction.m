clear all;
clc;
namelist         = {'../data/normal','../data/ruptured_eardrum','../data/otitis_media','../data/earwax_blockage'};
REPEAT_LEN       = 20;
[reference, ~]   = audioread('40_seconds_chirp.wav');
Ydata            = [];
subjs            = 0;
%%
names    = [];
Xdata    = [];
% empty_list = [7,8,14,16,17,33,42,54,56,57,63,64,69,72,74,84,85,86, 87, 88,90,91,92];
empty_list = [];
for i    = 1:length(namelist)
filelist = file_read(namelist{i});
for f    = 1:length(filelist)
subjs    = subjs + 1;
sub_var1         = [filelist{f}];
disp(sub_var1)
if ismember(subjs,empty_list)
    names = [names,sub_var1];
end
if ~ismember(subjs,empty_list)
[audio, Fs]      = audioread(sub_var1);
[~, l]           = xcorr( audio(1:Fs), reference(1:Fs) );
[maxvalue, I]    = max(f);
offset           = I;
%% find start point
audio_start          = audio(offset:end);
mm = 0;
%
for m = 1:REPEAT_LEN
mm = mm +1;
audio_channel       = audio_start(Fs*m+1: Fs * 1 + Fs*m, 1);
ref_input           = reference(Fs*m+1:Fs*m+length(audio_channel));
Y                   = transfer_function(ref_input,audio_channel,Fs);

fs     = Fs;
nfs    = 8192;
wind   = hann(0.4*fs);
nov    = 0.39*fs;

tf_seq = 0;

tf_seq = tf_seq + 1;
[txy,tfs] = tfestimate(ref_input, audio_channel,wind,nov,nfs,fs);
txy_norm  = movmean(20*log10(abs(txy)),128);
txy_norm1 = normalize(txy_norm(1:1400),'range',[0 1]);


fft_feat = fft_feature(audio_channel);
% 
% name      = ['csv/' 'train_' 'sub_' num2str(subjs) '_seq_' num2str(m) '_' num2str(i) '.csv'];
% writematrix([txy_norm1',fft_feat], name)
% 
% labelname = ['csv/' 'label_' 'sub_' num2str(subjs) '_seq_' num2str(m) '_' num2str(i) '.csv'];
% writematrix(i, labelname)

Xdata = [Xdata;txy_norm1',fft_feat];
Ydata = [Ydata;i];
end
end
end
end

writematrix(Xdata, 'Xdata.csv')
writematrix(Ydata, 'Ydata.csv')

%%