function V = getvalue(Q)
% GETVALUE returns the V-function from the Q-matrix.
% 
% You have to implement this function yourself. It is not necessary to loop
% in order to do this, and looping will be much slower than using matrix
% operations. It's possible to implement this in one line of code.

V = max(Q,[],3);

end

