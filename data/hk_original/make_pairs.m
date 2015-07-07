function [P, Y] = make_pairs(I, S, group, nMatching, nNonMatching)
% MAKE_PAIRS build pairs of elements across sessions
i = 1;
P = zeros(0, 2, 'uint32');
Y = false(0,1);
inGp = ismember(I, group);
for id = group'
    fromS1 = find(I == id & S == 1);
    fromS2 = find(I == id & S == 2);
    nonpeers = find(I ~= id & inGp);

    P(i:i + numel(fromS1) * nMatching - 1, 1) = ...
    reshape(repmat(fromS1', nMatching, 1), [], 1);
    P(i:i + numel(fromS1) * nMatching - 1, 2) = ...
    fromS2(randi(numel(fromS2), numel(fromS1)*nMatching, 1));
    Y(i:i + numel(fromS1) * nMatching - 1) = true;

    i = i + numel(fromS1) * nMatching;

    P(i:i + numel(fromS1) * nNonMatching - 1, 1) = ...
    reshape(repmat(fromS1', nNonMatching, 1), 1, []);
    P(i:i + numel(fromS1) * nNonMatching - 1, 2) = ...
    nonpeers(randi(numel(nonpeers), numel(fromS1)*nNonMatching, 1));
    Y(i:i + numel(fromS1) * nNonMatching - 1) = false;
    
    i = i + numel(fromS1) * nNonMatching;
end
end