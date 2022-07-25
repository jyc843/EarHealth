function fft_feat = fft_feature(audio_channel)
fftY               = abs(fft(audio_channel));
Y1                 = fftY(20:6000);
% from 20 hz to 6 kHz to find acoustic dip
max_freq_distance  = 650;
Y                  = movmean(Y1(max_freq_distance:length(Y1)-max_freq_distance),64);
[peaks, locs1]     = findpeaks(-Y,'MinPeakWidth',15,'Npeaks',15,'minpeakdistance',max_freq_distance,'SortStr','descend');

% normalization
val1_min           = min(Y1);
[val1_max,~]       = max(Y1);
Y1                 = (Y1 - val1_min)/(val1_max - val1_min);
[~,peak_loc]       = max(peaks);
locs               = locs1(peak_loc)+max_freq_distance;

train_freq_mean      = mean(Y1(locs-max_freq_distance:locs+max_freq_distance-1),1);
train_freq_dev       = std(Y1(locs-max_freq_distance:locs+max_freq_distance-1),0,1);
train_freq_skewness  = skewness(Y1(locs-max_freq_distance:locs+max_freq_distance-1),1,1);
train_freq_kurtosis  = kurtosis(Y1(locs-max_freq_distance:locs+max_freq_distance-1),1,1);
train_freq_crest     = 20*log(max(abs(Y1(locs-max_freq_distance:locs+max_freq_distance-1)),[],1)./train_freq_dev);
train_freq_flatness  = geomean((abs(Y1(locs-max_freq_distance:locs+max_freq_distance-1)).^2),1)./mean((abs(Y1(locs-max_freq_distance:locs+max_freq_distance-1)).^2),1);
train_freq_autocorr  = autocorr(Y1(locs-max_freq_distance:locs+max_freq_distance-1));
train_freq_meancross = mean_crossings(Y1(locs-max_freq_distance:locs+max_freq_distance-1));
fft_feat = [train_freq_mean,train_freq_dev,train_freq_skewness,train_freq_kurtosis,train_freq_crest,train_freq_flatness,train_freq_autocorr',train_freq_meancross];
end
%%