using LinearAlgebra
using Statistics
using Random

const n_components = 3 # how many ICA components

const max_iter = 200

const tolerance = 0.000001

function negentropy_logcosh(x::Real, alpha::Real=1.0)::Tuple{Real, Real}
    x *= alpha
    first_deriv = tanh(x)
    second_deriv = alpha * (1 - first_deriv^2)

    return first_deriv, second_deriv
end


function fastICA(M::Matrix{<:Real}, n_components, max_iter=200, tolerance=1e-6)
M_cent = M .- transpose(mean.(eachcol(M))) # centered observation matrix - columns with each one's mean subtracted

M_cent_sq = transpose(M_cent) * M_cent

# Eigendecomposition used here for whitening, SVD not implemented (yet).
eigdec = eigen(M_cent_sq)

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

whitened = whitening_mat * transpose(M_cent) * sqrt(size(M_cent, 1))

# XXX normally this will be a (n_components, n_components) random normally-distributed matrix 
#random_matr = [0.2519036  1.01005105 -0.74699474
#            -0.22713996  0.84246727 -0.00284533
#            0.05731927 -0.79115478  1.51150747]
random_matr = randn(n_components, n_components)

# Numpy format (w_init variable within FastICA._fit_transform()):
# array([[ 0.2519036 ,  1.01005105, -0.74699474],
#        [-0.22713996,  0.84246727, -0.00284533],
#        [ 0.05731927, -0.79115478,  1.51150747]])

W = zeros((n_components, n_components))

gd1 = Array{eltype(whitened)}(undef, size(whitened, 2))
gd2 = Array{eltype(whitened)}(undef, size(whitened, 2))
iter_counts = zeros(Int, n_components)


for i in 1:n_components
    w_row = copy(random_matr[i, :]) # random weights row vector
    w_row = w_row / norm(w_row)

    k::Int = 0
    for k in 1:max_iter
        w_rowX = transpose(w_row) * whitened # Python: np.dot(w_row.T, X) 

        # Marcin recommends using for loops with memory preallocation like in C
        for p in 1:size(whitened, 2)
            gd1[p], gd2[p] = negentropy_logcosh(w_rowX[p])
        end
        
        #show(size(mean(gderivs2)))
        #println(typeof(mean(gderivs2)))
        #println(size(w_row * 5))

        # Single component extraction, step 2 in Wiki "FastICA" article
        #println(size(mean(whitened .* transpose(gderivs1), dims=2)))
        #println(size(w_row * mean(gderivs2)))
        w1 = mean(whitened .* transpose(gd1), dims=2)[:, 1] - w_row * mean(gd2)
        
        #w2 = w1 - (transpose(w1) * transpose(W[1:i, :]) * W[1:i, :])
        w2 = w1' - transpose(w1) * transpose(W[1:i, :]) * W[1:i, :]
        w3 = w2 / norm(w2) # normalize to unit length

        lim = abs(abs(sum(w3 .* transpose(w_row))) - 1)
        w_row = transpose(w3)

        if lim < tolerance
            break
        end
        #show(w1)
    end

    iter_counts[i] = k
    W[i, :] = w_row
end

#show(W)

sources = transpose(W * whitening_mat * M_cent') # _fastica.py line 663
                  # Should have dimensions (2000, 3)

# We implement only for "arbitrary-variance" whitening in sklearn,
# so the section under "unit-variance" is skipped.

# W should be the unmixing matrix, consider returning it somehow.
return sources
end
