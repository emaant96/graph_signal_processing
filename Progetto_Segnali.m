
clear
clc

run gspbox/gsp_start

tol = 1e-15;
tol2 = 1e-7;
% RAW_REF 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv';
% REF 'https://github.com/pcm-dpc/COVID-19/blob/master/dati-province/dpc-covid19-ita-province-latest.csv';
% RAW_REF_TOT 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
data_tot = readtable('data/dati_nazionali.csv');
date = datetime(2020,02,24) : datetime(2022,04,20);
figure;plot(date,data_tot.totale_positivi);
ylabel('totale positivi');
title('Andamento Covid in Italia');

%%
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
I = eye(num_nodi);
G = gsp_graph(A,[x_coord y_coord]);
figure;gsp_plot_graph(G);title('Grafo delle province');

%% graph and signal

init = 670; % giorno 2021-12-23 670
interest = 700; % giorno 2022-01-22 700

init_cases = data_filt((init-1) * num_nodi + 1:init*num_nodi,:);
max_cases = data_filt((interest-1) * num_nodi + 1:interest*num_nodi,:);
interest_cases = max_cases;
interest_cases.totale_casi = (interest_cases.totale_casi - init_cases.totale_casi) ./ pop.popolazione * 100;

s = interest_cases.totale_casi;

param.climits = [0,max(s)];
figure;gsp_plot_signal(G,s,param);title('Nuovi casi covid dal 23/12 al 01/22 (%pop)');

%% computation Laplacian Eigenvectors Eigenvalues

d = sum(A,2);
L = diag(d) - A;

[U,v] = eig(L);
v = diag(v);
%% computation Frequency Profile on Graph (GFT)

f = U' * s;
f_mod = sqrt(f.^2);
figure;plot(v,f_mod,'r.-');
xlabel('eigenvalues');ylabel('coefficient');
title('Graph Frequency profile');

%% filtering

num_freq = 47;
h = [ones(num_freq,1) ; zeros(num_nodi - num_freq,1)];

%% direct computation of H from eigenvectors

s_filt = U * diag(h) * U' * s;
f = U' * s_filt;
f_mod = sqrt(f.^2);
figure;plot(v,f_mod,'r.-');
xlabel('eigenvalues');ylabel('coefficient');
title('Graph Frequency profile');

figure;
gsp_plot_signal(G,s_filt,param);title('Grafo infetti filtrati');

difference = s - s_filt;
param1.bar = 1;
param1.bar_width = 3;
figure;
gsp_plot_signal(G,difference,param1);
figure;
stem(1:1:102,difference,'filled');ylim([-max(s)/5,max(s)/5]);title('Differenza segnale filtrato e non filtrato');

%% simulation of distribuited computation of approximated filter H with Eigenvalues and Laplacian
tic
p = 16;
V = zeros(num_nodi,p + 1);

for i = 0:p
    V(:,i+1) = v.^i;
end

V = vpa(V);

alpha = pinv(V) * h;
z_i = s;
s_filt2 = zeros(num_nodi,1);

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

error = mse(s_filt2 - s_filt);

disp("error distribuited calc approximation of filter: " + string(error))

param.climits = [0,max(s)];
figure;gsp_plot_signal(G,s_filt2,param);title('Totale nuovi casi filtrati filtro approssimato')

%% sampling and reconstruction

ds = [0,0,1,1,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,0,0,1,0,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,0,1,1,1,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1,0,1,1,1,0,1,0,1,0];
nodi_considerati = sum(ds)
s_camp = diag(ds)*s_filt;
param.climits = [0,max(s)];
figure;gsp_plot_signal(G,s_camp,param);title('Totale nuovi casi campionati')
Ef = diag(h);
Bf = U*Ef*U';

% Sampling Theorem

cDs = I - diag(ds);

sv = sqrt(max(eig(vpa(cDs*Bf))));
disp("largest singular value: " + string(sv))

Pf = zeros(num_freq, length(s));
j = 1;
for i = 1:length(h)
    if(h(i) == 1)
        Pf(j,i) = 1;
        j = j + 1;
    end
end
Uf = U*Pf';

s_interp = Uf*((Uf'*diag(ds)*Uf)\Uf')*s_camp;
error = mse(s_filt - s_interp);
disp("error interpolation PsUf pseudoinverse approch: " + string(error))

s_interp = (I - cDs*Bf)\s_camp;
error = sum(abs(s_filt - s_interp));
disp("error interpolation Q approch: " + string(error))

%% iterative version

s_interp1 = s_camp;
error = zeros(100,1);
for p = 1:100
    s_interp1 = s_camp + cDs*Bf*s_interp1;
    error(p) = mse(s_filt - s_interp1);
end
figure;plot(error);xlabel('iteration');ylabel('MSE');
disp("error interpolation iterative version algoritm approximation: " + string(error(length(error))))

%% reconstruction with eigenvector and eigenvalue of BDB

[U1,v1] = eig(Bf*diag(ds)*Bf);
v1 = diag(v1);
s_interp2 = zeros(length(s),1);

for i = 1:45
    s_interp2 = s_interp2 + (s_camp'*U1(:,i)/v1(i))*U1(:,i);
end

error = mse(s_filt - s_interp2);
disp("error interpolation eigenvector and eigenvalue of BDB approximation: " + string(error))

%% print error for privinces
figure;gsp_plot_signal(G,s_interp,param);title('Totale nuovi casi segnale ricostruito')
difference = s - s_interp;
figure;plot(difference,'g-');ylim([-max(s),max(s)]);title('Differenza segnale campionato/ricostruto e segnale originale');


errore = abs(s - s_interp);
[err_prov,idx] = maxk(errore,10);
err_prov
interest_cases(idx,:).denominazione_provincia


