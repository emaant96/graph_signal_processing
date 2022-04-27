
clear
clc
tol = 1e-15 * 100000000;

numb_filt = 47;
p = 51;
%% graph and signal

interest_cases = readtable('data/signal_1223-0122.csv');
num_nodi = length(interest_cases.totale_casi);
s = interest_cases.totale_casi;

x_coord = interest_cases.long;
y_coord = interest_cases.lat;

%% computation Laplacian Eigenvectors Eigenvalues
A = readmatrix('data/adj_matrix.csv');

% G = gsp_graph(A,[x_coord y_coord]);

d = sum(A,2);
L = diag(d) - A;
[U,v] = eig(L);
v = diag(v);

%% filtering

h = [ones(num_nodi-numb_filt,1) ; zeros(numb_filt,1)];

%% direct computation of H from eigenvectors

s_filt = U * diag(h) * U' * s;

%% simulation of distribuited computation of approximated filter H

V = zeros(num_nodi,p + 1);

for i = 0:p
    V(:,i+1) = v.^i;
end
V = vpa(V);

alpha = pinv(V) * h;

z_i = s;
s_filt2 = zeros(num_nodi,1);
I = eye(num_nodi);

for n = 1:length(alpha)
    z_temp = z_i;
    for i = 1:num_nodi
        % computation of new value node i by diffusion of neigbors values
        neib_value = 0;
        if(n == 1)
            z_i(i) = z_temp(i);
        else
            for j = 1:num_nodi
                if(L(i,j) ~= 0)  % only neigbors have value Lij > 0
                    neib_value = neib_value + L(i,j) * z_temp(j);
                end
            end
            z_i(i) = neib_value;
        end
    end
    s_filt2 =  s_filt2 + z_i .* alpha(n);
end

error = sum(abs(s_filt2 - s_filt));
disp("error distribuited calc approximation of filter: " + string(error))

%% sampling and reconstruction

%ds = [0,0,1,1,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,0,1,1,1,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,1,0,1,0,1,0];
%47
Ef = diag(h);
Bf = U*Ef*U';
Pf = zeros(num_nodi-numb_filt, length(s));
j = 1;
for i = 1:length(h)
    if(h(i) == 1)
        Pf(j,i) = 1;
        j = j + 1;
    end
end

Uf = U*Pf';

cvx_begin
    variable w(102)
    minimize(lambda_sum_largest(Uf'*diag(w)*Uf,47))
    subject to
    zeros(102,1) <= w <= ones(102,1)
    sum(w) == 55
cvx_end

ds = w;
cDs = I - diag(ds);
cDsBf = cDs * Bf;
s_camp = diag(ds)*s_filt;


% Sampling Theorem



sv = sqrt(max(eig(vpa(cDsBf))));
disp("largest singular value: " + string(sv))


s_interp = Uf*((Uf'*diag(ds)*Uf)\Uf')*s_camp;
error = sum(abs(s_filt - s_interp));
disp("error interpolation PsUf pseudoinverse approch: " + string(error))

s_interp = vpa((I - cDsBf)\s_camp);
error = sum(abs(s_filt - s_interp));
disp("error interpolation Q approch: " + string(error))
%% iterative version

s_interp1 = s_camp;

for k = 1:100000
    s_interp1 = s_camp + cDsBf*s_interp1;
end
error = sum(abs(s_filt - s_interp1));
disp("error interpolation iterative version algoritm approximation: " + string(error))

%% reconstruction with eigenvector and eigenvalue of BDB

[U1,v1] = eig(Bf*diag(ds)*Bf);
v1 = diag(v1);
s_interp2 = zeros(length(s),1);

for i = 1:55
        s_interp2 = s_interp2 + (s_camp'*U1(:,i)/v1(i))*U1(:,i);
    
end

error = sum(abs(s_filt - s_interp2));
disp("error interpolation eigenvector and eigenvalue of BDB approximation: " + string(error))
%% 


