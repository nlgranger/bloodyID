function [P, Y] = make_pairs(I, S, group, nMatching, nNonMatching)
% MAKE_PAIRS build pairs of elements across sessions
i = 1;
P = zeros(0, 2, 'uint32');
for id = group
    fromS1 = find(I == id & S == 1);
    fromS2 = find(I == id & S == 2);
    nonpeers = find(I ~= id);

    P(i:i + numel(fromS1) * nMatching - 1, 1) = ...
    reshape(repmat(fromS1', nMatching, 1), 1, []);
    P(i:i + numel(fromS1) * nMatching - 1, 2) = ...
    fromS2(randi(numel(fromS2), 1, numel(fromS1)*nMatching));
    Y(i:i + numel(fromS1) * nMatching) = true;

    i = i + numel(fromS1) * nMatching;

    P(i:i + numel(fromS1) * nNonMatching - 1, 1) = ...
    reshape(repmat(fromS1, nNonMatching, 1), 1, []);
    P(i:i + numel(fromS1) * nNonMatching - 1, 2) = ...
    nonpeers(randi(numel(nonpeers), 1, numel(fromS1)*nNonMatching));
    i = i + numel(fromS1) * nNonMatching;
end
end