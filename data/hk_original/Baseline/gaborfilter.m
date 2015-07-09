function G = gaborfilter(D, S, l, theta)
x = repmat((0:D(1)-1) - D(1)/2, 1, D(2));
y = repmat((0:D(2)-1) - D(2)/2, D(1), 1);
R = [cos(theta) sin(theta); -sin(theta) cos(theta)];
rotated = R * [reshape(x, 1, []); reshape(y, 1, [])];
x = reshape(rotated(1,:), D);
y = reshape(rotated(2,:), D);

G = exp(-0.5 * ((x .^ 2) / (S(1)^2) + (y .^ 2) / (S(2)^2))) ...
    .* cos(2 * pi / l * x);
end
