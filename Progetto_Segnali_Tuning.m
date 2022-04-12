
clear
clc
tol = 1e-15 * 100000000;
run gspbox/gsp_start


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
U(abs(U) < tol) = 0;
v(abs(v) < tol) = 0;

%% filtering

numb_filt = 50;
h = [ones(num_nodi-numb_filt,1) ; zeros(numb_filt,1)];

%% direct computation of H from eigenvectors

s_filt = U * diag(h) * U' * s;

%% simulation of distribuited computation of approximated filter H
p = 25;

V = zeros(num_nodi,p + 1);

for i = 0:p
    V(:,i+1) = v.^i;
end
V = vpa(V);
rank(V)
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

ds = [0,0,1,1,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,0,1,1,1,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,1,0,1,0,1,0];

s_camp = diag(ds)*s_filt;
Ef = diag(h);
Bf = U*Ef*U';

% Sampling Theorem

cDs = I - diag(ds);

sv = svds(cDs*Bf,1,'largest');
disp("largest singular value: " + string(sv))
Pf = zeros(num_nodi-numb_filt, length(s));

j = 1;
for i = 1:length(h)
    if(h(i) == 1)
        Pf(j,i) = 1;
        j = j + 1;
    end
end

Uf = U*Pf';
s_interp = Uf*((Uf'*diag(ds)*Uf)\Uf')*s_camp;
error = sum(abs(s_filt - s_interp));
disp("error interpolation pseudoinverse approch: " + string(error))
%% iterative version

s_interp1 = s_camp;
for p = 1:100000
    s_interp1 = s_camp + cDs*Bf*s_interp1;
end
error = sum(abs(s_filt - s_interp1));
disp("error interpolation iterative version algoritm approximation: " + string(error))
%% reconstruction with eigenvector and eigenvalue of BDB

[U1,v1] = eig(Bf*diag(ds)*Bf);
v1 = diag(v1);
U1(abs(U1) < tol) = 0;
v1(abs(v1) < tol) = 0;
U1 = vpa(U1);
v1 = vpa(v1);
s_interp2 = zeros(length(s),1);

for i = 1:length(v1)
    if(v1(i) ~= 0)
        s_interp2 = s_interp2 + (s_camp'*U1(:,i)/v1(i))*U1(:,i);
    end
end

error = sum(abs(s_filt - s_interp2));
disp("error interpolation eigenvector and eigenvalue of BDB approximation: " + string(error))



