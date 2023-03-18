using LinearAlgebra
using Statistics
using NPZ


const n_components = 3 # how many ICA components
npz = npzread("scikit_fastICA_BSS/sin_sign_saw.npz")["arr_0"]
#print(typeofnpz)
npz_cent = npz .- transpose(mean.(eachcol(npz)))

npz_cent_sq = transpose(npz_cent) * npz_cent

eigdec = eigen(npz_cent_sq)

if any(eigdec.values .< eps(1.0))
    @warn "An eigenvalue has a very low magnitude, which could be a sign of problems." eigdec.values
end

eigvals_sqrt = sqrt.(eigdec.values)
eigvals_sqrt_reverse_order = sortperm(eigvals_sqrt, rev=true)
eigvals_rev_order = eigvals_sqrt[eigvals_sqrt_reverse_order]
eigvecs_rev_order = eigdec.vectors[:, eigvals_sqrt_reverse_order]
# display(eigvals_rev_order)
# display(eigdec.values)
# display(eigvecs_rev_order)
eigvecs_sign_corrected = eigvecs_rev_order .* transpose(sign.(eigvecs_rev_order[1,:]))

# Whitening matrix as in eq. 6.33 (p. 140) in "Independent Component Analysis" (2001) book
whitening_mat = (eigvecs_sign_corrected ./ eigvals_rev_order')'[1:n_components, :] # there shouldn't be any complex values here, so using ' operator
# The sklearn version truncates the rows of the whitening matrix to n_components

whitened = whitening_mat * transpose(npz_cent) * sqrt(size(npz_cent, 1))

# XXX normally this will be a (n_components, n_components) random normally-distributed matrix 
random_matr = [0.2519036  1.01005105 -0.74699474
            -0.22713996  0.84246727 -0.00284533
            0.05731927 -0.79115478  1.51150747]

# Numpy format (w_init variable within FastICA._fit_transform()):
# array([[ 0.2519036 ,  1.01005105, -0.74699474],
#        [-0.22713996,  0.84246727, -0.00284533],
#        [ 0.05731927, -0.79115478,  1.51150747]])

W = zeros((n_components, n_components))

for i in 1:n_components
    
end