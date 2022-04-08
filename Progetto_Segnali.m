
clear
clc
tol = 1e-15;
% RAW_REF url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv';
% REF https://github.com/pcm-dpc/COVID-19/blob/master/dati-province/dpc-covid19-ita-province-latest.csv
run gspbox/gsp_start

data = readtable('data/data.csv');
pop = readtable('data/popolazione_province_connesso.csv');
num_nodi = length(pop.popolazione);
column  = ["data", "denominazione_regione", "denominazione_provincia", "lat", "long", "totale_casi"];
data_filt = rmmissing(data(:,column));
data_filt = data_filt(string(data_filt.denominazione_regione) ~= 'Sardegna',:);

x_coord = data_filt(1:num_nodi,:).long;
y_coord = data_filt(1:num_nodi,:).lat;

A = zeros(num_nodi,num_nodi);

for i = 1:num_nodi
    dis_array = zeros(1,num_nodi);
    for j = 1:num_nodi 
        dis_array(j) = sqrt((y_coord(j) - y_coord(i))^2 + (x_coord(j) - x_coord(i))^2);
    end
    
    % take only points at distance < 1.5
    idx = find(dis_array < 1.5);
    values = dis_array(idx); 
    
    % take only 7 nearest
    [~,idxv] = mink(values,7);
    idx = idx(idxv);
    
    % connect nodes nearest 
    A(i,idx(idx~=i)) = 1;
end

% make undirected the graph 
A = A + A';
A(A == 2) = 1;

G = gsp_graph(A,[x_coord y_coord]);
figure;gsp_plot_graph(G);title('Grafo delle province');

%% graph and signal

init = 670; % 2021-12-23
interest = 700; % 2022-01-22

init_cases = data_filt((init-1) * num_nodi + 1:init*num_nodi,:);
max_cases = data_filt((interest-1) * num_nodi + 1:interest*num_nodi,:);
interest_cases = max_cases;
interest_cases.totale_casi = (interest_cases.totale_casi - init_cases.totale_casi) ./ pop.popolazione * 100;

s = interest_cases.totale_casi;

param.climits = [0,13];
figure;gsp_plot_signal(G,s,param);title('Infetti totali fino al giorno massimo');

%% computation Laplacian Eigenvectors Eigenvalues

d = sum(A,2);
L = diag(d) - A;

[U,v] = eig(L);
v = diag(v);
U(abs(U) < tol) = 0;
v(abs(v) < tol) = 0;
%% computation Frequency Profile on Graph (GFT)

f = U' * s;
f_mod = sqrt(f.^2);
figure;plot(v,f_mod,'r.-');
xlabel('eigenvalues');ylabel('coefficient');
title('Graph Frequency profile');

%% filtering

numb_filt = 60;
h = [ones(num_nodi-numb_filt,1) ; zeros(numb_filt,1)];

%% direct computation of H from eigenvectors

s_filt = U * diag(h) * U' * s;

xlabel('nodi');ylabel('infetti');
figure;gsp_plot_signal(G,s_filt,param);title('Totale nuovi casi filtrati');

difference = s - s_filt;
figure;plot(difference,'g-');ylim([-max(s),max(s)]);title('Differenza segnale filtrato e non filtrato');

%% approximation with Eigenvalues and Laplacian
tic
p = 6;

V = zeros(num_nodi,p + 1);

for i = 0:p
    V(:,i+1) = v.^i;
end

alpha = ((V'*V) \ V') * h;
H = zeros(num_nodi);

for i = 0:p
    H = H + alpha(i+1)*L^i;
end

s_filt1 = H * s;
toc
error = sum(abs(s_filt1 - s_filt));
disp("error computation approximated of filter with Laplacian Matrix powers: " + string(error))
figure;gsp_plot_signal(G,s_filt1);title('Totale nuovi casi filtrati filtro approssimato')

%% simulation of distribuited computation of approximated filter H
tic
p = 6;

V = zeros(num_nodi,p + 1);

for i = 0:p
    V(:,i+1) = v.^i;
end

alpha = ((V'*V) \ V') * h;
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
toc

error = sum(abs(s_filt2 - s_filt));
disp("error distribuited calc approximation of filter: " + string(error))
figure;gsp_plot_signal(G,s_filt1);title('Totale nuovi casi filtrati filtro approssimato')

%% sampling and reconstruction

% tablenuova = groupfilter(interest_cases,'denominazione_regione',@(x) ismember(x,maxk(x,int16(length(x)/2))))
ds = [0,0,1,1,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,0,1,1,1,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,1,0,1,0,1,0];

s_camp = diag(ds)*s_filt;
figure;gsp_plot_signal(G,s_camp);title('Totale nuovi casi filtrati filtro approssimato')
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
U1(abs(U1) < tol*100000000) = 0;
v1(abs(v1) < tol*100000000) = 0;
s_interp2 = zeros(length(s),1);

for i = 1:length(v1)
    if(v1(i) ~= 0)
        s_interp2 = s_interp2 + (s_camp'*U1(:,i)/v1(i))*U1(:,i);
    end
end

error = sum(abs(s_filt - s_interp2));
disp("error interpolation eigenvector and eigenvalue of BDB approximation: " + string(error))

%% print error for privinces
figure;gsp_plot_signal(G,s_interp);title('Totale nuovi casi filtrati filtro approssimato')
difference = s - s_interp;
figure;plot(difference,'g-');ylim([-max(s),max(s)]);title('Differenza segnale campionato/ricostruto e segnale originale');
errore_medio = abs(s-s_interp);
[v,idx] = maxk(errore_medio,10);
v
interest_cases(idx,:).denominazione_provincia



