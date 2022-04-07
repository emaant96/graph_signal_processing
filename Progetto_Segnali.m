
clear
clc
tol = 1e-10;
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
A = A + A.';
A(A == 2) = 1;

G = gsp_graph(A,[x_coord y_coord]);
figure;gsp_plot_graph(G);title('Grafo delle province');

%% graph and signal

init = 670; % 2021-12-23
interest = 700; % 2022-01-22

init_cases = data_filt((init-1) * num_nodi + 1:init*num_nodi,:).totale_casi;

s = data_filt((interest-1) * num_nodi + 1:interest*num_nodi,:).totale_casi;
s = s - init_cases;
s = s ./ pop.popolazione * 100;

param.climits = [0,13];
figure;gsp_plot_signal(G,s,param);title('Variazione casi');

%% computation Laplacian Eigenvectors Eigenvalues

d = sum(A,2);
L = diag(d) - A;

[U,v] = eig(L);
v = diag(v);

%% computation Frequency Profile on Graph (GFT)

f = U.' * s;
f_mod = sqrt(f.^2);
figure;plot(v,f_mod,'r.-');
xlabel('eigenvalues');ylabel('coefficient');
title('Graph Frequency profile');

%% filtering

numb_filt = 20;
h = [ones(num_nodi-numb_filt,1) ; zeros(numb_filt,1)];

%% direct computation of H from eigenvectors

s_filt = U * diag(h) * U.' * s;

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

alpha = ((V.'*V) \ V.') * h;
H = zeros(num_nodi);

for i = 0:p
    H = H + alpha(i+1)*L^i;
end

s_filt1 = H * s;
toc
error = sum(abs(s_filt1 - s_filt))

figure;gsp_plot_signal(G,s_filt1);title('Totale nuovi casi filtrati filtro approssimato')

%% simulation of distribuited computation of approximated filter H
tic
p = 6;

V = zeros(num_nodi,p + 1);

for i = 0:p
    V(:,i+1) = v.^i;
end

alpha = ((V.'*V) \ V.') * h;
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

error = sum(abs(s_filt2 - s_filt))
figure;gsp_plot_signal(G,s_filt1);title('Totale nuovi casi filtrati filtro approssimato')

%% sampling and reconstruction

Ds = zeros(length(s));
freq_camp = 2;

for i = 1:num_nodi
    if (mod(i,freq_camp) == 0)
        Ds(i,i) = 1;
    end
end

s_camp = Ds*s;
f = U.'*s_camp;
figure;plot(f,'r-');ylim([-20,80])
xlabel('eigenvalues');ylabel('coefficient');
title('Graph Frequency profile');

Bf = U*Ef*U.';
Bf(abs(Bf) < tol) = 0;
e = eig(Bf*Ds*Bf);
if(max(e) - 1 < tol)
    disp("signal on graph is both vertex and frequency limited")
end

% Sampling Theorem

cDs = I - Ds;

eigv = eig(cDs*Bf);
sqrt(max(eigv))



