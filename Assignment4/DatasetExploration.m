clear
clc
close all
%% Data Loading

dataset = 'example1.dat';
%dataset = 'example2.dat';
E = csvread(dataset);

col1 = E(:,1);
col2 = E(:,2);
max_ids = max(max(col1,col2));
As= sparse(col1, col2, 1, max_ids, max_ids); 
A = full(As);

%% Plot the Visualization of the Dataset

figure;
G = graph(A);
p = plot(G,'layout','force');
title(['Graph Dataset ' dataset]);

%% Plot the Sparsity Pattern

figure;
spy(A,'r');
title(['Sparsity Pattern' dataset]);