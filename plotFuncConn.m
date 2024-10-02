load('distribution_pairwise_vinograd_nair.mat')

% all_dist_mean_x1 represents pairwise interactions between x1->x1 neurons
% all_dist_mean_x2 represents pairwise interactions between x1->x2 neurons

% plot distributions as in Vinograd*, Nair*
figure
edges = [0:0.1:2];
histogram(all_dist_mean_x1,edges,'Normalization','count')
hold on
histogram(all_dist_mean_x2,edges,'Normalization','count')
xlim([-0.1, 2.1])
xlabel('average z-scored activity ')
xline(0.7)

x1_desity = nnz(all_dist_mean_x1>0.7)/length(all_dist_mean_x1);

% plot fit distribution
figure
edges = [0:0.1:2];
histfit(all_dist_mean_x1,20,'kernel')
hold on
histfit(all_dist_mean_x2,20,'kernel')
xlim([-0.1, 2.1])
xlabel('average z-scored activity ')
xline(0.7)