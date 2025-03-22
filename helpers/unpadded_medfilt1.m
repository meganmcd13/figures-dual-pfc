function x_filt = unpadded_medfilt1(x,n)
    % filter trace x using median filter of size n
    % with truncated edges (only use valid points for filter convolution)
    
    need_transpose = size(x,2)==1;
    if need_transpose, x = x'; end
    x_filt = medfilt1(x,n,'truncate'); % use truncate to handle edges
    if need_transpose, x_filt = x_filt'; end
    
end

