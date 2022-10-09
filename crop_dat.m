function [ Data,fade_vector ] = crop_dat( Data,image_size,fade_vector )
%crop_dat(Data(2,n_long), image_size = [Y,X])
%   Data(2,n_long), image_size = [Y,X]

Limit = not(logical(sum(Data < 1) | Data(2,:) > image_size(2)  | Data(1,:) > image_size(1)));        %Pixels position less than 1 (outside image to left and top)
Data = Data(:,Limit);
if not(isscalar(fade_vector))
    fade_vector = fade_vector(Limit);
end

%X_limit
% Limit = not(logical(Data(2,:) > image_size(2)));        %Pixels outside to right
% Data = Data(:,Limit);
% if not(isscalar(fade_vector))
%     fade_vector = fade_vector(Limit);
% end
% 
% %Y_limit
% Limit = not(logical(Data(1,:) > image_size(1)));        %Pixels outside to bottom side
% Data = Data(:,Limit);
% if not(isscalar(fade_vector))
%     fade_vector = fade_vector(Limit);
% end

end

