clear %all
rng(1);

filename = 'oscillofun_01.wav';
wav_noise = 0.00025;                % add noise to analaog wave signal for extra realism
image_filename = 'Output\Animation_';
use_gpu = 1;
polar = 1;                          % render as X-Y?
start_frame = 1;                    % default 1
image_size = [2160,3840];           % in pixels (later supersampling is done)
frame_rate = 59.94;
gamma = 1;                          % lower value is higher brightness
image_AA = 2;                       % supersampling of image resolution, integer 1 is disable
brightness = 0.01;                 % times of overexposure
line_width = 0.6;                   % width of line after supersampling etc.:
line_blur_spread = 16;              % width of blur/spread thickness after supersampling etc.:
HSV = [0.45,0.8,1];                 % Line color (Saturation should be less than 0.9)
upsampling_total_points = 12e6;      % supersampling (float >= 1)
fade_with_time = 1;                 % logical bolean
fading_frames = ceil(frame_rate*4);      % fade goes back this many frames in time
faded_intensity = 0.667;              % Drop in intensity between active beam to after-glow

enabledust = 0;
dustmap = 'dusty.png';
dust_brightness = 16;
dust_blur_size = 160;

X_scale = 1.3/16*9;                     % oscillofun = 1.3/16*9
Y_scale = 1.3;                          % oscillofun = 1.3


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Reading file, timing start')
tic
[WAV_dat,Fs] = audioread(filename);

zero_padding = zeros(round(Fs*fading_frames/(frame_rate))+1,2)   -   2*polar;   %dot starts outside frame
zero_padding_2 = zeros(ceil(0.04*round(Fs*fading_frames/(frame_rate)))+1,2);    %dot in center for some time in beginning
WAV_dat = [zero_padding;zero_padding_2;WAV_dat;zero_padding];
        WAV_dat = WAV_dat + normrnd(0, wav_noise, size(WAV_dat));           %% Add randomness

clear zero_padding zero_padding_2

if polar == 0
    WAV_dat(:,2) = -((WAV_dat(:,1)+WAV_dat(:,2))/2);
    t = linspace(0,length(WAV_dat)/Fs,length(WAV_dat));
    WAV_dat(:,1) = -sawtooth(2*pi*frame_rate*t);
end

frames_start_points = round(0:Fs/frame_rate:length(WAV_dat))+1;
frame_time = round(fading_frames*Fs/(frame_rate));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Color map generation
RGB = hsv2rgb(HSV);
map_index = linspace(0,1,65536);
multfactor=20;
map = [tanh(map_index*RGB(1)*multfactor);tanh(map_index*RGB(2)*multfactor);tanh(map_index*RGB(3)*multfactor)]';
map = min(map,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Blurring filter shader
range_filter = ceil(10*line_blur_spread)*2+1;                %defines the size of the gaussian filter to apply for point spreading/blurring
E = fspecial('gaussian', range_filter, line_width);               % Create the gaussian filter for small dot in centre

x = -(range_filter-1)/2:(range_filter-1)/2;
F = zeros(range_filter);
G = zeros(range_filter);

for y = x
    F(y+(range_filter-1)/2+1,:) = exp( -sqrt( x.^2 + y.^2 )/(line_blur_spread*0.2) );        %Line widening
    G(y+(range_filter-1)/2+1,:) = exp( -sqrt( x.^2 + y.^2 )/(range_filter/16) );             %Blur around line
end
F = F/sum(sum(F));                                              % Smaller blur closer to center
G = G/sum(sum(G));                                              % Long tail of the blurring (faint spreading)

Dot_intens = [1 0.4*2 1*2];
H = E*Dot_intens(1) + F*Dot_intens(2) + G*Dot_intens(3);                % Weighting of line, E center spot, F small glow near spot, G long faint glow
Dot_scale = 1/sum(sum(H));
H = H*Dot_scale;

if 0      % Plotting
    semilogy(H(:,(size(F,1)+1)/2))
    hold on
    semilogy(Dot_scale*E(:,(size(F,1)+1)/2)*Dot_intens(1),':')
    semilogy(Dot_scale*G(:,(size(F,1)+1)/2)*Dot_intens(2),':')
    semilogy(Dot_scale*F(:,(size(F,1)+1)/2)*Dot_intens(3),':')
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fade with time
if fade_with_time
    log_fade_1 =     10.^linspace(-2*fading_frames/5, log10(faded_intensity), round(upsampling_total_points*(fading_frames-1)/fading_frames));          % Fast fade from high intensity to very low for short time after exposure
    log_fade_2 =     10.^linspace(-6, -1.6, round(upsampling_total_points*(fading_frames-1)/fading_frames));                                              % Slow fade from medium intensity to low intensity for long time
    full_intensity = ones(1,round(upsampling_total_points/fading_frames));
    fade_vector =    [(log_fade_1+log_fade_2), full_intensity];
    if 0      % Plotting
        semilogy(fade_vector)
        hold on
        return
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dust image paramters and map loading
if enabledust
  disp('Load dust image')
  if use_gpu
    dustmap = gpuArray(imresize(single(imread(dustmap)),image_size));       % load dustmap and resize of nessecary
  else
    dustmap = imresize(single(imread(dustmap)),image_size);       % load dustmap and resize of nessecary
  end
  dustmap = dustmap(:,:,1)/max(max(max(dustmap)));
  D = fspecial('gaussian',dust_blur_size*7,dust_blur_size);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Start main loop')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time_elapsed = toc;
end_frame = length(frames_start_points) - ceil(fading_frames);
for i = start_frame:end_frame    % +1

    Y_dat = WAV_dat(frames_start_points(i):frames_start_points(i)+frame_time,:);
    X_dat = -Y_dat(:,1)';
    Y_dat = Y_dat(:,2)';

    [X_dat,Y_dat] = supsamp_func(X_dat,Y_dat,upsampling_total_points);
    [image] = PPP_func3(X_dat,Y_dat,X_scale,Y_scale,image_size,image_AA,brightness,H,fade_vector,use_gpu);

    % Add dust
    if enabledust
        dustimage = dustmap.*imfilter(image,D)*dust_brightness;
        if use_gpu
          wait(gpuDevice());
          dustimage = gray2ind(gather(dustimage),65535);            %rescale to 0-65535 values
        else
          dustimage = gray2ind(dustimage,65535);            %rescale to 0-65535 values
        end
        dustimage = uint16( ind2rgb(dustimage,map)*65535 );                                     %convert indexed colormapped image to RGB
        write_dust_filename = [image_filename,'dust_',num2str(i),'.tif'];
        imwrite(dustimage,write_dust_filename,'tif','compression','LZW');         %Write image file6
    end

    image = mat2gray(max(image,0),[0 1]);         %cap values between 0 and 1
    %image = image.^gamma;                   %curve gamma adjustment
    if use_gpu
      wait(gpuDevice());
      image = gray2ind(gather(image),65535);            %rescale to 0-65535 values
    else
      image = gray2ind(image,65535);            %rescale to 0-65535 values
    end

    image = uint16( ind2rgb(image,map)*65535 );                                     %convert indexed colormapped image to RGB

    write_filename = [image_filename,num2str(i),'.tif'];
    imwrite(image,write_filename,'tif','compression','LZW');         %Write image file

    frame_timing = toc-time_elapsed;
    time_elapsed = toc;
    remaining_time = (end_frame-i)*frame_timing;

    disp([num2str(time_elapsed,'Elapsed time:%8.2f s '),num2str(frame_timing,'(%5.2f s)'),', Writing:"', ...
        write_filename,num2str(end_frame,'", Total:%6i'),' (remaining: ' , ...
        datestr(remaining_time/86400, 'HH:MM:SS'),')'])
    %pause(0.001)
end

disp('--- Done ---')
