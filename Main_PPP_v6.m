clear %all

tic
use_gpu = 1;
filename = 'AM_modulation.wav';
image_filename = 'Output\Image_1';
image_size = [1080,1920];           %in pixels (later supersampling is done)
image_AA = 3;                       % supersampling of image resolution, integer 1 is disable
upsampling_total_points = 20e6;     % times of supersampling
brightness = 0.0002;                  % times of overexposure
line_width = 0.6;                   % Width of line after supersampling etc.:
line_blur_spread = 16;              % Width of blur/spread thickness after supersampling etc.:
gamma = 0.8;                        % Gamma, lower value gives brighter image
HSV = [0.45,0.8,1];                 % Line color (Saturation should be less than 0.9)
fade_vector = [];                         % No fading with time

X_scale = [];
Y_scale = 0.8;

%%%%%%%%%%%%%%%%%%%%%%% Reading File %%%%%%%%%%%%%%%%%%%%%%
disp('Reading file')
[WAV_dat,Fs] = audioread(filename);
Y_dat = WAV_dat(1:end,1)';
X_dat = linspace(0,1,numel(Y_dat));

%%%%%%%%%%%%%%%%%%%%%%%%  AM testing %%%%%%%%%%%%%%%%%%%%%%
%X_dat = linspace(0,1,10000);
%ramp = logspace(1.2,2.6,10000);
%Y_dat = sin(X_dat*(2*pi).*ramp).*((sin(X_dat*2*pi*1.5).^2+0.04));
%Y_dat = Y_dat./max(abs(Y_dat));

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

Dot_intens = [1 0.4 1];
H = E*Dot_intens(1) + F*Dot_intens(2) + G*Dot_intens(3);                % Weighting of line, E center spot, F small glow near spot, G long faint glow
Dot_scale = 1/sum(sum(H));
H = H*Dot_scale;


[X_dat,Y_dat] = supsamp_func(X_dat,Y_dat,upsampling_total_points);

disp('Running PPP function')
[image] = PPP_func3(X_dat,Y_dat,X_scale,Y_scale,image_size,image_AA,brightness,H,fade_vector,use_gpu);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write file
write_filename = [image_filename,'.tif'];
disp(['Writing: ',write_filename])


image = mat2gray(max(image,0),[0 1]);         %cap values between 0 and 1
image = image.^gamma;                   %curve gamma adjustment
if use_gpu
  wait(gpuDevice());
  image = gray2ind(gather(image),65535);            %rescale to 0-65535 values
else
  image = gray2ind(image,65535);            %rescale to 0-65535 values
end
image = uint16( ind2rgb(image,map)*65535 );                                     %convert indexed colormapped image to RGB

imwrite(image,write_filename,'tif','compression','LZW');         %Write image file

disp('--- Done ---')
toc
