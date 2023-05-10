function nnInputMat=create_matrix(input)
m = 2;
k = 9;
inputLayerDim = 2*m +k*m;
nnInputMat = zeros(length(input),inputLayerDim);
for i = 2:(length(input))
   for j = 0:(m-1)
		nnInputMat(i,j+1) = real(input(i-j));
		nnInputMat(i,j+m+1) = imag(input(i-j));
    end
	for j = (2*m):((2+k)*m - 1)
		power = (floor(j/m) - 1);
        nnInputMat(i,j+1) = abs(input(i-mod(j,m)))^power;
    end
end
end
