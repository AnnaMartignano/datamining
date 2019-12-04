clear
clc
close all
%% Dataset Loading

%filename = 'example1.dat';
filename = 'example2.dat';
data = csvread(filename);

%% 

%{
Given a set of points S = {s1, ... Sn} we want to cluster into k clusters.

Algorithm:
1. Form the affinity matrix A,  Aij = exp((-||s_i - s_j||^2)/2(sigma)^2) 
   for i not j and Aii = 0.
2. Define D to be the diagonal matrix whose (i,i)-element is the sum of A's
   i-th row and construct the matrix L = D^(-1/2)AD^(-1/2).
3. Find x1, x2, .., xk the k largest eigenvectors of L (chosen to be
   orthogonal to each other in the case of repeated eigenvalues), and form
   the matrix X = [x1x2..xk] by stacking the eigenvectors in columns.
4. Form the matrix Y from X by renormalizing each of X's rows to have unit
   length (i.e. Yij = Xij/(sumj (Xij)^2)^(1/2)).
5. Treating each row of Y as a point in Rk, cluster them into k clusters
   via K-means or any other algorithm.
6. Finally, assign the original point si to the cluster j if and only if
   row i of the matrix Y was assigned to cluster j.

The scaling parameter sigma^2 controls how rapidly the affinity Aij falls
off with the distance between si and sj. 
%}

% hyperparameter to tune k and sigma
sigma_range = 1 : 0.5 : 5;

min_sumd = inf;
for sigma = sigma_range
    [orders, idx, G, eig_val, sumdist, k] = spectral_clustering(data, sigma);
    if sum(sumdist) < min_sumd
        min_sumd = sum(sumdist);
        sigma_opt = sigma;
        orders_opt = orders;
        idx_opt = idx;
        k_opt = k;
        eig_val_opt = eig_val;
    end
end

%% Plot Original and Clustered Graph
p = plot(G,'layout','force');
title(['Graph Dataset ' filename]);

figure,
plot(eig_val_opt);
title(['Sorted eigenvalues ' filename]);

size(idx_opt)
idx_opt;
figure;
hold on;
h = plot(G);
highlight(h,find(idx_opt==1),'NodeColor','r')
highlight(h,find(idx_opt==2),'NodeColor','g')
highlight(h,find(idx_opt==3),'NodeColor','b')
highlight(h,find(idx_opt==4),'NodeColor','c')
title([filename ' ,' num2str(k_opt) ' clusters with sigma ' num2str(sigma_opt) ]);

% Step 6
[~, r_orders] = sort(orders_opt, 'descend');
clusters = idx_opt(r_orders);


%% Function Definition

function [orders, idx, G, eig_val, sumdist, k] = spectral_clustering(data, sigma)
    % Step 1
    %A = compute_A(data, sigma);
    col1 = data(:,1);
    col2 = data(:,2);    
    G = graph( col1, col2 );
    Ad = adjacency(G);
    A = full(Ad);

    % Step 2
    D = diag(sum(A, 2));
    L = single(D^(-0.5)*A*D^(-0.5)); % converts matrix to single-precision otherwise they are not symmetric

    % Step 3
    [v, d] = eig(L);
    [eig_val, orders] = sort(diag(d), 'descend');
    k = find_cluster_size(eig_val);
    eig_vec = v(:, orders);
    X = eig_vec(:, 1 : k);

    % Step 4
    Y = normr(X);

    % Step 5
    [idx, C, sumdist] = kmeans(Y, k);
end

function A = compute_A(x, sigma)
    % squareform returns a symmetric matrix where Z(i,j) corresponds to the
    % pairwise distance between observations i and j.
    si_sj_dist = squareform(pdist(x));  

    % .^ -> element-wise power and ./ -> element-wise division
    A = exp(-(si_sj_dist.^2)./(2*sigma^2)); % according to the formula
    A = A - diag(diag(A));      % to set Aii = 0 
end
function k = find_cluster_size(eig_val)
    diffs = eig_val(1:end-1) - eig_val(2:end);
    [top, k] = max(diffs);
end