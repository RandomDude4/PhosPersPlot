function [ image ] = PPP_func3(X_dat,Y_dat,X_scale,Y_scale,image_size,image_AA,brightness,G,fade_vector,use_gpu)
    %PPP_func

    image_size = image_size*image_AA;


    if isempty(X_scale)
        X_dat = (X_dat-min(X_dat));
        X_dat = X_dat*2/max(X_dat)-1;
        X_scale = 0.95;
    end
    X_dat = round((X_dat*X_scale+1)*(image_size(2)-1)/2+1);                    %scale to full image height and round to even pixel
    Y_dat = round((Y_dat*Y_scale+1)*(image_size(1)-1)/2+1);

    Data = [Y_dat;X_dat];
    clear Y_dat X_dat

    if numel(fade_vector) > 1
        %log_fade =          faded_intensity*( linspace(0.1,1,round(length(Data(1,:))*(fading_frames-1)/fading_frames)).^128 + faded_intensity*10.^linspace(-4,-1,round(length(Data(1,:))*(fading_frames-1)/fading_frames)));
        %full_intensity =    ones(1,round(length(Data(1,:))/fading_frames));
        %fade_vector =       [log_fade, full_intensity];
        [Data,fade_vector] = crop_dat(Data,image_size,fade_vector);                                        %Remove data outside image dimensions
        image = single( accumarray(Data',fade_vector,image_size) );
    else
        Data = crop_dat(Data,image_size,0);                                                                %Remove data outside image dimensions
        image = single( accumarray(Data',1,image_size) );                                                   %Raytrace all pixels
    end
    clear Data

    image = imresize(image,1/image_AA,'bilinear');                  % Has to be performed in CPU since bilinear does not work in GPU (cubic interp generates artifacts)
    if use_gpu
      image = gpuArray(image);
    end

    image = imfilter(image,G);              %# Filter it
    image = image*brightness*image_AA^2;               %apply brightness, correct for downscaling after AA
end
