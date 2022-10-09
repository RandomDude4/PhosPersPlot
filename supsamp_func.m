function [X,Y] = supsamp_func(X,Y,upsampling_total_points)
%supsampl_func 
%   Upsamples data to include mor points with spline
%   if X is different length than Y, X vector is created
        
if length(Y) >= upsampling_total_points         %if data has enough samples
    upsampling_total_points = length(Y);    
    if length(X) ~= length(Y)
        X = linspace(-1,1,length(Y));
    end
else                                            %data upsampling neccesary
    if length(X) ~= length(Y)
        %X = linspace(-1,1,upsampling_total_points);
        X = linspace(1,length(Y),upsampling_total_points);
        Y = interp1(Y,X,'spline');
        max_X = max(X-1);
        X = ((X-1)-max_X/2)/(max_X/2);
    else
        t = linspace(1,length(Y),upsampling_total_points);
        X = interp1(X,t,'spline');
        Y = interp1(Y,t,'spline');
    end
end


end

