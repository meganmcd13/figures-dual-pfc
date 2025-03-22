function [ C ] = normCoincidence( X )
% Input:
%    - X: spike train matrix (nChannels x nT)
%
% Output:
%    - C: row-wise percentage of coincident spikes  (wrt total spikes of that neuron)
%
    
    s = X*X';
    
    n_spikes = sum(X,2);
    
    for i_row = 1:size(s,1)
        s(i_row,:) = s(i_row,:)./n_spikes(i_row);
    end

    C = s;
    C(diag(diag(true(size(C))))) = nan;
    
end



