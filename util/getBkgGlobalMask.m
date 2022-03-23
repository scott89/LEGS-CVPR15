function background_global_mask = getBkgGlobalMask(width, height)
%  bg_width = ceil(max(width, height)*15/400);
bg_width = 15;
background_global_mask = false(height, width, 5);
background_global_mask(:, :, 1) = true; % global
background_global_mask(1:bg_width, :, 2) = true; % top
background_global_mask(end-bg_width+1:end, :, 3) = true; %bottom
background_global_mask(:, 1:bg_width, 4) = true; % left
background_global_mask(:, end-bg_width+1:end, 5) = true; % right

