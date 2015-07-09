function I = imshift(I, shift, varargin)
    bg = 0;
    if ~isempty(varargin)
        bg = varargin{1};
    end
    vz      = shift(1);
    hz      = shift(2);
    [h, w]  = size(I);
    
    I = [bg*ones(vz, w); ...
         I(max(1, 1-vz):min(h-vz, h), :); ...
         bg*ones(-vz, w)];
    I = [bg*ones(h, hz) I(:, max(1, 1-hz):min(w, w-hz)) bg*ones(h, -hz)];
end